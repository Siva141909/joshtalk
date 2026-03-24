# =============================================================================
# Q3: Hindi Spelling Error Detection on ~1,77,000 Unique Words
# Platform: CPU (Kaggle or local)
# No paid APIs — uses open Hindi resources + rule-based signals
# =============================================================================

# ── CELL 1: Install ──────────────────────────────────────────────────────────
# !pip install -q requests indic-nlp-library

# ── CELL 2: Imports ──────────────────────────────────────────────────────────
import re, json, csv, unicodedata, urllib.request, zipfile, io, os
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

import os as _os
if _os.path.exists("/kaggle/working"):
    OUT_DIR = Path("/kaggle/working/output")
else:
    OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STEP 1: Build word list
# If you have the 1,77,000 word file from Josh Talks, load it directly.
# Otherwise we extract from our transcription data.
# =============================================================================

# ── CELL 3: Load / Extract Unique Words ──────────────────────────────────────
def extract_words_from_transcriptions(manifest_csv: str) -> Counter:
    """Extract all word tokens from reference transcriptions."""
    word_counts = Counter()
    df = pd.read_csv(manifest_csv)
    for text in df['text'].dropna():
        for w in text.split():
            w_clean = re.sub(r'[।\.\,\!\?\:\;\"\'०-९0-9]', '', w).strip()
            if len(w_clean) > 0:
                word_counts[w_clean] += 1
    return word_counts

# Load Josh Talks word list if provided as a text file (one word per line)
WORD_LIST_FILE = str(Path(__file__).parent / "unique_words.txt")  # optional: one word per line

if os.path.exists(WORD_LIST_FILE):
    with open(WORD_LIST_FILE, encoding='utf-8') as f:
        unique_words = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(unique_words):,} words from provided word list")
else:
    # Fall back: extract from our transcriptions
    word_counts = extract_words_from_transcriptions(str(Path(__file__).parent / "Datasets" / "segments_manifest.csv"))
    unique_words = list(word_counts.keys())
    print(f"Extracted {len(unique_words):,} unique words from transcriptions")

# =============================================================================
# STEP 2: Build reference Hindi vocabulary
# Sources (all free):
#   1. Download AI4Bharat IndicNLP Hindi word frequency list
#   2. IndoWordNet lookup via pyiwn (optional, heavier)
# =============================================================================

# ── CELL 4: Download Hindi Reference Vocabulary ──────────────────────────────
def download_hindi_wordlist() -> set:
    """Try multiple sources for Hindi word list."""
    urls = [
        # Leipzig Hindi frequency list (100k words)
        "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/hi/hi_50k.txt",
        # Backup: smaller curated list
        "https://raw.githubusercontent.com/lorenbrichter/Words/master/Words/hi.txt",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                content = r.read().decode('utf-8')
            words = set()
            for line in content.splitlines():
                # Format: "word frequency" or just "word"
                word = line.strip().split()[0] if line.strip() else ''
                if word and not word.isdigit():
                    words.add(word)
            if words:
                print(f"Downloaded Hindi word list: {len(words):,} entries")
                return words
        except Exception as e:
            print(f"  Source failed ({e}), trying next...")
    print("All sources failed — using transcription vocabulary only")
    # Build vocab from our own transcription data as fallback
    word_counts = extract_words_from_transcriptions(
        str(Path(__file__).parent / "Datasets" / "segments_manifest.csv")
    )
    # High-frequency words in our data are likely correct
    return {w for w, c in word_counts.items() if c >= 2}


hi_vocab = download_hindi_wordlist()
print(f"Reference vocabulary size: {len(hi_vocab):,}")

# =============================================================================
# STEP 3: Spelling Classification Pipeline
# Multi-signal approach — each signal contributes to confidence
# =============================================================================

# ── CELL 5: Signal Functions ─────────────────────────────────────────────────

def is_valid_devanagari(word: str) -> bool:
    """Check if word contains only Devanagari + standard diacritics."""
    for ch in word:
        cp = ord(ch)
        if not (0x0900 <= cp <= 0x097F or  # Devanagari block
                0x0900 <= cp <= 0x0963 or  # vowels/consonants
                ch in ' '):
            return False
    return True

# Known English words transcribed in Devanagari (correct per guidelines)
DEVANAGARI_ENGLISH_KNOWN = {
    'इंटरव्यू','जॉब','मोबाइल','कंप्यूटर','इंटरनेट','ऑनलाइन','ऑफलाइन',
    'प्रोजेक्ट','मीटिंग','प्रेजेंटेशन','ऑफिस','सैलरी','टीम','मैनेजर',
    'बॉस','कोर्स','डिग्री','एग्जाम','रिजल्ट','मार्क्स','फोन',
    'वेबसाइट','ऐप','एप','डिजिटल','सोशल','मीडिया','प्लेटफॉर्म',
    'सॉफ्टवेयर','हार्डवेयर','स्कूल','कॉलेज','यूनिवर्सिटी',
    'ओके','हेलो','थैंक्यू','सॉरी','प्लीज',
    'एक्चुअली','बेसिकली','नॉर्मल','सीरियस','प्रॉपर','स्पेशल',
    'एरिया','ट्रेनिंग','इंटर्नशिप','रिसर्च','पेपर','रिपोर्ट',
}

# Common misspelling patterns in Hindi
MISSPELLING_SIGNALS = [
    # Repeated same consonant without halant (uncommon in Hindi)
    (r'(.)\1\1', "triple character repetition"),
    # Halant at end of word (unusual)
    (r'्$', "halant at word end"),
    # Matra without preceding consonant
    (r'^[ािीुूेैोौं]', "matra at start"),
    # Two consecutive full vowels (not diacritics)
    (r'[अआइईउऊएऐओऔ]{3,}', "3+ consecutive vowels"),
    # Anusvara followed by another anusvara
    (r'ंं', "double anusvara"),
    # Visarga in non-standard positions
    (r'ः[^।\s]', "visarga mid-word"),
]

def misspelling_pattern_score(word: str) -> tuple[int, list[str]]:
    """Returns (penalty_count, list of triggered patterns)."""
    triggered = []
    for pattern, name in MISSPELLING_SIGNALS:
        if re.search(pattern, word):
            triggered.append(name)
    return len(triggered), triggered

def unicode_normal_check(word: str) -> bool:
    """Word should be in NFC normalized form; if not, likely a transcription artifact."""
    return unicodedata.normalize('NFC', word) == word

# ── CELL 6: Classifier ───────────────────────────────────────────────────────
def classify_word(word: str) -> dict:
    """
    Classify a word as correctly or incorrectly spelled.
    Returns: {word, label, confidence, reasons}
    """
    reasons = []
    correct_signals = 0
    incorrect_signals = 0

    # Signal 1: Non-Devanagari (Roman/numeric etc.)
    if not is_valid_devanagari(word):
        # Roman script word or mix — could be transcription error
        if re.match(r'^[a-zA-Z]+$', word):
            return {'word': word, 'label': 'incorrect spelling',
                    'confidence': 'high',
                    'reason': 'Roman script in Devanagari transcription'}
        if re.match(r'^[0-9]+$', word):
            return {'word': word, 'label': 'incorrect spelling',
                    'confidence': 'high',
                    'reason': 'Digit string — should be written as word'}
        reasons.append("non-standard characters")
        incorrect_signals += 1

    # Signal 2: Known Devanagari-English word (correct per guidelines)
    if word in DEVANAGARI_ENGLISH_KNOWN:
        return {'word': word, 'label': 'correct spelling',
                'confidence': 'high',
                'reason': 'Known English loanword correctly transcribed in Devanagari'}

    # Signal 3: In reference Hindi vocabulary
    if word in hi_vocab:
        correct_signals += 2
        reasons.append("in reference vocabulary")

    # Signal 4: Misspelling pattern signals
    penalty, patterns = misspelling_pattern_score(word)
    if penalty > 0:
        incorrect_signals += penalty
        reasons.append(f"misspelling patterns: {', '.join(patterns)}")

    # Signal 5: Unicode normalization
    if not unicode_normal_check(word):
        incorrect_signals += 1
        reasons.append("not NFC normalized")

    # Signal 6: Very short (1 char) — likely transcription noise
    if len(word) == 1 and word not in 'एकदोतीनचारपांचछसातआठनौ':
        incorrect_signals += 1
        reasons.append("single character, likely noise")

    # Signal 7: Very long word (>20 chars) — unusual in Hindi
    if len(word) > 20:
        incorrect_signals += 1
        reasons.append("unusually long word")

    # Decision
    net_score = correct_signals - incorrect_signals

    if net_score >= 2:
        label, confidence = 'correct spelling', 'high'
    elif net_score == 1:
        label, confidence = 'correct spelling', 'medium'
    elif net_score == 0:
        label, confidence = 'correct spelling', 'low'
    elif net_score == -1:
        label, confidence = 'incorrect spelling', 'low'
    else:
        label, confidence = 'incorrect spelling', 'high' if net_score <= -2 else 'medium'

    # If not in vocab and no other signals, it's ambiguous → low confidence
    if word not in hi_vocab and penalty == 0 and not any('character' in r for r in reasons):
        confidence = 'low'
        if label == 'correct spelling':
            reasons.append("OOV — not in reference vocab (may be valid proper noun/loanword)")

    return {
        'word'      : word,
        'label'     : label,
        'confidence': confidence,
        'reason'    : '; '.join(reasons) if reasons else 'in vocabulary' if word in hi_vocab else 'no signals'
    }

# ── CELL 7: Classify All Words ───────────────────────────────────────────────
print(f"Classifying {len(unique_words):,} words...")
results = [classify_word(w) for w in unique_words]
result_df = pd.DataFrame(results)

# Summary
correct_df   = result_df[result_df['label'] == 'correct spelling']
incorrect_df = result_df[result_df['label'] == 'incorrect spelling']

print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
print(f"Total unique words         : {len(result_df):,}")
print(f"Correctly spelled          : {len(correct_df):,}")
print(f"Incorrectly spelled        : {len(incorrect_df):,}")
print(f"\nConfidence breakdown (correct):")
print(result_df[result_df['label']=='correct spelling']['confidence'].value_counts().to_string())
print(f"\nConfidence breakdown (incorrect):")
print(result_df[result_df['label']=='incorrect spelling']['confidence'].value_counts().to_string())

# ── CELL 8: Low-Confidence Review (40–50 words) ───────────────────────────────
low_conf = result_df[result_df['confidence'] == 'low'].sample(
    min(50, result_df[result_df['confidence']=='low'].shape[0]), random_state=42
)

print(f"\n=== LOW CONFIDENCE REVIEW ({len(low_conf)} words) ===")
print("Manual review required — flagged for human verification:\n")
for i, row in low_conf.head(50).iterrows():
    print(f"  [{i:5d}] {row['word']:<25} → {row['label']:<22} | {row['reason']}")

# After manual review, note findings:
print("""
\nExpected findings from low-confidence review:
  - Proper nouns (place names, person names) — often OOV but correct
  - Rare/archaic Hindi words — OOV but correct
  - English loanwords not in whitelist — may be correct per guidelines
  - Compound words / sandhi forms — OOV but valid
  → These are the categories where our system is unreliable
""")

# ── CELL 9: Unreliable Categories ────────────────────────────────────────────
print("=== UNRELIABLE WORD CATEGORIES ===")
print("""
CAT-1: PROPER NOUNS (names, cities, brands)
  Problem: Not in standard Hindi dictionaries → flagged as OOV → low confidence
  Examples: "नरेंद्र", "मुंबई", "जियो", "हिंदुस्तान"
  Why unreliable: Our classifier has no named-entity awareness.
  Fix: Build/use a proper noun gazetteer.

CAT-2: ENGLISH LOANWORDS IN DEVANAGARI (per transcription guidelines)
  Problem: Words like "इंटरव्यू", "सॉफ्टवेयर" are CORRECT per Josh Talks guidelines
           but absent from Hindi dictionaries → classified as incorrect/low-confidence
  Examples: "प्रेजेंटेशन", "डेडलाइन", "वर्कशॉप"
  Why unreliable: Whitelist is incomplete; new loanwords emerge constantly
  Fix: Expand loanword whitelist or use a transliteration model.
""")

# ── CELL 10: Save Google Sheet–ready output ───────────────────────────────────
output_path = OUT_DIR / "q3_spelling_classification.csv"
result_df[['word', 'label', 'confidence', 'reason']].to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
print(f"Final count of CORRECTLY SPELLED unique words: {len(correct_df):,}")
print("\nUpload this CSV to Google Sheets — 2 columns needed:")
print("  Column A: word")
print("  Column B: correct spelling / incorrect spelling")
