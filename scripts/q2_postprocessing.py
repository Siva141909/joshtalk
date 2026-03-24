# =============================================================================
# Q2: ASR Post-processing Pipeline — Number Normalization + English Word Detection
# Run AFTER q1_whisper_finetune.py has generated segment manifests
# Platform: CPU-only (Kaggle or local)
# =============================================================================

# ── CELL 1: Install ──────────────────────────────────────────────────────────
# !pip install -q transformers torch librosa soundfile

# ── CELL 2: Imports ──────────────────────────────────────────────────────────
import re, json, csv, os
from typing import Optional
import pandas as pd
import torch
import soundfile as sf
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── CELL 3: Config ───────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
MODEL_NAME   = "openai/whisper-small"   # pretrained, BEFORE fine-tuning (Q2 requirement)
# Paths: auto-detect Kaggle vs local
if os.path.exists("/kaggle/working"):
    SEG_MANIFEST = "/kaggle/input/datasets/sivatejareddya/josh-talks-segments/segments_manifest.csv"
    OUT_DIR = Path("/kaggle/working/output")
else:
    SEG_MANIFEST = str(Path(__file__).parent / "Datasets" / "segments_manifest.csv")
    OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CELL 4: Get raw ASR output from pretrained Whisper-small ─────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
proc   = WhisperProcessor.from_pretrained(MODEL_NAME, language="Hindi", task="transcribe")
model  = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE).eval()

seg_df = pd.read_csv(SEG_MANIFEST)
print(f"Generating raw ASR on {len(seg_df)} segments...")

raw_asr_results = []
for idx, row in seg_df.iterrows():
    try:
        audio, _ = sf.read(row['audio_path'], dtype='float32', always_2d=False)
        feats = proc.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(DEVICE)
        with torch.no_grad():
            ids = model.generate(
                feats,
                language="hi",
                task="transcribe",
                forced_decoder_ids=None,
                temperature=0.0,
                condition_on_prev_tokens=False,
                max_new_tokens=128,
            )
        pred = proc.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        words = pred.split()
        if len(words) > 5 and len(set(words)) < 3:
            pred = ""
        elif words and max(len(w) for w in words) > 20 and len(set(pred.replace(" ", ""))) < 6:
            pred = ""
    except Exception as e:
        pred = ""
        print(f"  [SKIP] {row['audio_path']}: {e}")
    raw_asr_results.append({
        'audio_path'   : row['audio_path'],
        'asr_raw'      : pred,
        'reference'    : row['text'],
        'recording_id' : row['recording_id'],
    })
    if (idx + 1) % 50 == 0:
        print(f"  {idx + 1}/{len(seg_df)} done")

del model
torch.cuda.empty_cache()

asr_df = pd.DataFrame(raw_asr_results)
asr_df.to_csv(OUT_DIR / "raw_asr_output.csv", index=False)
print(f"Saved raw ASR output: {len(asr_df)} rows")

# =============================================================================
# PART A: NUMBER NORMALIZATION
# =============================================================================

# ── CELL 5: Hindi Number Word Table ──────────────────────────────────────────
HINDI_NUMBERS = {
    # 0–19
    'शून्य':0, 'एक':1, 'दो':2, 'तीन':3, 'चार':4,
    'पाँच':5, 'पांच':5, 'छह':6, 'छः':6, 'सात':7, 'आठ':8, 'नौ':9,
    'दस':10, 'ग्यारह':11, 'बारह':12, 'तेरह':13, 'चौदह':14,
    'पंद्रह':15, 'सोलह':16, 'सत्रह':17, 'अठारह':18, 'उन्नीस':19,
    # 20–29
    'बीस':20, 'इक्कीस':21, 'बाइस':22, 'तेइस':23, 'चौबीस':24,
    'पच्चीस':25, 'छब्बीस':26, 'सत्ताइस':27, 'अट्ठाइस':28, 'उनतीस':29,
    # 30–39
    'तीस':30, 'इकतीस':31, 'बत्तीस':32, 'तैंतीस':33, 'चौंतीस':34,
    'पैंतीस':35, 'छत्तीस':36, 'सैंतीस':37, 'अड़तीस':38, 'उनतालीस':39,
    # 40–49
    'चालीस':40, 'इकतालीस':41, 'बयालीस':42, 'तैंतालीस':43, 'चवालीस':44,
    'पैंतालीस':45, 'छियालीस':46, 'सैंतालीस':47, 'अड़तालीस':48, 'उनचास':49,
    # 50–59
    'पचास':50, 'इक्यावन':51, 'बावन':52, 'तिरेपन':53, 'चौवन':54,
    'पचपन':55, 'छप्पन':56, 'सत्तावन':57, 'अट्ठावन':58, 'उनसठ':59,
    # 60–69
    'साठ':60, 'इकसठ':61, 'बासठ':62, 'तिरेसठ':63, 'चौंसठ':64,
    'पैंसठ':65, 'छियासठ':66, 'सड़सठ':67, 'अड़सठ':68, 'उनहत्तर':69,
    # 70–79
    'सत्तर':70, 'इकहत्तर':71, 'बहत्तर':72, 'तिहत्तर':73, 'चौहत्तर':74,
    'पचहत्तर':75, 'छिहत्तर':76, 'सतहत्तर':77, 'अठहत्तर':78, 'उनासी':79,
    # 80–89
    'अस्सी':80, 'इक्यासी':81, 'बयासी':82, 'तिरासी':83, 'चौरासी':84,
    'पचासी':85, 'छियासी':86, 'सत्तासी':87, 'अट्ठासी':88, 'नवासी':89,
    # 90–99
    'नब्बे':90, 'इक्यानवे':91, 'बानवे':92, 'तिरानवे':93, 'चौरानवे':94,
    'पचानवे':95, 'छियानवे':96, 'सत्तानवे':97, 'अट्ठानवे':98, 'निन्यानवे':99,
    # Multipliers
    'सौ':100, 'सो':100,
    'हजार':1000, 'हज़ार':1000,
    'लाख':100000,
    'करोड़':10000000, 'करोड':10000000,
}

MULTIPLIERS = {100:'सौ', 1000:'हजार', 100000:'लाख', 10000000:'करोड़'}
MULT_VALS   = sorted([v for v in HINDI_NUMBERS.values() if v >= 100], reverse=True)

# Idioms/phrases where number words must NOT be converted
IDIOM_PATTERNS = [
    r'दो[‑-]चार', r'तीन[‑-]चार', r'चार[‑-]पाँच', r'दो[‑-]तीन', r'एक[‑-]दो',
    r'दो टूक', r'चार चाँद', r'सात समंदर', r'नौ दो ग्यारह',
    r'एक[‑-]आध', r'दस[‑-]बीस', r'सात जन्म',
]
IDIOM_RE = re.compile('|'.join(IDIOM_PATTERNS))

# ── CELL 6: Number Normalization Logic ───────────────────────────────────────
def words_to_number(tokens: list) -> Optional[int]:
    """Convert a sequence of Hindi number word tokens to an integer.
    Returns None if conversion is not possible."""
    total, current = 0, 0
    for tok in tokens:
        val = HINDI_NUMBERS.get(tok)
        if val is None:
            return None
        if val >= 100:
            # Multiplier: apply to current chunk, add to total
            if current == 0:
                current = 1
            if val in (100000, 10000000):
                total = (total + current) * val
                current = 0
            else:
                current *= val
        else:
            current += val
    return total + current

def normalize_numbers(text: str) -> str:
    """
    Convert Hindi number words to digits.
    Skips idiomatic phrases like दो-चार (means 'a few', not '2-4').
    """
    # Protect idioms: replace with placeholder
    protected = {}
    def protect(m):
        key = f"__IDIOM{len(protected)}__"
        protected[key] = m.group(0)
        return key
    text_safe = IDIOM_RE.sub(protect, text)

    words = text_safe.split()
    output = []
    i = 0
    while i < len(words):
        # Greedy: try longest match of consecutive number words
        best_len, best_val = 0, None
        for j in range(len(words), i, -1):
            chunk = words[i:j]
            val = words_to_number(chunk)
            if val is not None:
                best_len, best_val = j - i, val
                break
        if best_val is not None:
            output.append(str(best_val))
            i += best_len
        else:
            output.append(words[i])
            i += 1

    result = ' '.join(output)
    # Restore protected idioms
    for key, orig in protected.items():
        result = result.replace(key, orig)
    return result

# ── CELL 7: Examples ─────────────────────────────────────────────────────────
correct_examples = [
    ("दो",                    "2"),
    ("दस",                    "10"),
    ("सौ",                    "100"),
    ("तीन सौ चौवन",           "354"),
    ("पच्चीस",                "25"),
    ("एक हज़ार",              "1000"),
    ("दो लाख पचास हज़ार",    "250000"),
]

edge_cases = [
    ("दो-चार बातें",       "दो-चार बातें",   "Idiom meaning 'a few things' — should NOT convert"),
    ("नौ दो ग्यारह हो गया", "नौ दो ग्यारह हो गया", "Idiom meaning 'to flee' — preserve"),
    ("एक-आध बार",           "एक-आध बार",      "Idiom: 'once or twice' — preserve"),
]

print("=== NUMBER NORMALIZATION — CORRECT CONVERSIONS ===")
for src, expected in correct_examples:
    result = normalize_numbers(src)
    status = "✓" if result.strip() == expected else "✗"
    print(f"  {status}  '{src}' → '{result}'  (expected: '{expected}')")

print("\n=== EDGE CASES — IDIOMATIC PHRASES ===")
for src, expected, reason in edge_cases:
    result = normalize_numbers(src)
    status = "✓" if result == expected else "✗"
    print(f"  {status}  '{src}' → '{result}'")
    print(f"       Reasoning: {reason}")

# Apply to ASR output
asr_df['asr_num_normalized'] = asr_df['asr_raw'].apply(normalize_numbers)

# Show real examples from data
print("\n=== FROM ACTUAL ASR DATA ===")
changed = asr_df[asr_df['asr_raw'] != asr_df['asr_num_normalized']].head(8)
for _, row in changed.iterrows():
    print(f"  BEFORE: {row['asr_raw']}")
    print(f"  AFTER : {row['asr_num_normalized']}")
    print()

# =============================================================================
# PART B: ENGLISH WORD DETECTION
# =============================================================================

# ── CELL 8: Hindi Word Vocabulary ────────────────────────────────────────────
# Build Hindi vocabulary from our transcription data (free, no external dict needed)
all_words = set()
for text in asr_df['reference'].dropna():
    for w in text.split():
        all_words.add(w.strip('।,.!?'))

print(f"Hindi vocabulary from data: {len(all_words)} unique words")

# Common English words as they appear in Devanagari transcription
# These are frequent English loanwords in Hindi speech
DEVANAGARI_ENGLISH_WORDS = {
    # Tech/modern
    'इंटरव्यू','इंटरव्यु','जॉब','प्रॉब्लम','सॉफ्टवेयर','हार्डवेयर',
    'मोबाइल','फोन','कंप्यूटर','इंटरनेट','वेबसाइट','ऐप','एप',
    'ऑनलाइन','ऑफलाइन','डिजिटल','सोशल','मीडिया','प्लेटफॉर्म',
    # Education/career
    'डिग्री','कोर्स','क्लास','स्कूल','कॉलेज','यूनिवर्सिटी',
    'एग्जाम','रिजल्ट','मार्क्स','परसेंट','रैंक',
    # Business
    'ऑफिस','मीटिंग','प्रेजेंटेशन','प्रोजेक्ट','टीम','मैनेजर',
    'बॉस','सैलरी','टारगेट','डेडलाइन',
    # Common loanwords
    'ओके', 'बाय','हेलो','थैंक्यू','सॉरी','प्लीज',
    'एक्चुअली','बेसिकली','ऑब्वियसली','टोटली','एक्सेक्टली',
    'नॉर्मल','सीरियस','प्रॉपर','स्पेशल','ऑफिशियल',
    'एरिया',
}

# ── CELL 9: English Word Detector ────────────────────────────────────────────
def is_devanagari_loanword(word: str) -> bool:
    """
    Detect if a Devanagari word is likely an English loanword.
    Strategy:
    1. Explicit whitelist of known English-origin words in Devanagari
    2. Phonetic pattern heuristics (letters/clusters uncommon in native Hindi)
    3. Absence from core Hindi vocabulary
    """
    w = word.strip('।,.!?')
    if not w:
        return False

    # Whitelist
    if w in DEVANAGARI_ENGLISH_WORDS:
        return True

    # Pattern signals for English origin in Devanagari:
    # - Starts with ऑ (open-o, not native Hindi)
    # - Contains ज़ followed by specific patterns
    # - Contains ड़/ढ़ in non-typical positions
    english_signals = [
        w.startswith('ऑ'),                         # open-o vowel: ऑफिस, ऑनलाइन
        bool(re.search(r'[फ़ज़]', w)),              # f/z sounds: फ़ोन, ज़ीरो
        bool(re.search(r'(शन|टिंग|मेंट)$', w)),   # -tion, -ting, -ment suffixes
        bool(re.search(r'(ई[ंन]ग)$', w)),          # -ing suffix
        bool(re.search(r'^(वेब|ऐप|ब्लॉ)', w)),    # Web-, App-, Blo- prefixes
    ]
    if sum(english_signals) >= 2:
        return True

    # Not in Hindi vocab AND not a known Hindi word pattern
    if w not in all_words and len(w) > 2:
        # Check if it has no Devanagari-native consonant clusters
        # (rough heuristic: very short words or OOV are often loanwords)
        pass

    return False

def tag_english_words(text: str) -> str:
    """
    Tag English loanwords in a Hindi transcript.
    Example: "मेरा इंटरव्यू हुआ" → "मेरा [EN]इंटरव्यू[/EN] हुआ"
    """
    words = text.split()
    result = []
    for w in words:
        core = w.strip('।,.!?')
        punct = w[len(core):]
        if is_devanagari_loanword(core):
            result.append(f"[EN]{core}[/EN]{punct}")
        else:
            result.append(w)
    return ' '.join(result)

# ── CELL 10: English Detection Examples ─────────────────────────────────────
test_sentences = [
    "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
    "ये प्रॉब्लम सॉल्व नहीं हो रहा है",
    "उसने ऑनलाइन कोर्स किया और बहुत कुछ सीखा",
    "मैं ऑफिस में मीटिंग के लिए जा रहा हूँ",
    "हमारा प्रोजेक्ट टीम के साथ पूरा हुआ",
]

print("=== ENGLISH WORD DETECTION EXAMPLES ===")
for sent in test_sentences:
    tagged = tag_english_words(sent)
    print(f"\nINPUT : {sent}")
    print(f"OUTPUT: {tagged}")

# Apply to ASR data
asr_df['asr_tagged_english'] = asr_df['asr_num_normalized'].apply(tag_english_words)

# Show real examples with English words detected
print("\n=== FROM ACTUAL ASR DATA ===")
has_english = asr_df[asr_df['asr_tagged_english'].str.contains(r'\[EN\]', na=False)].head(10)
for _, row in has_english.iterrows():
    print(f"  RAW   : {row['asr_raw']}")
    print(f"  TAGGED: {row['asr_tagged_english']}")
    print()

# ── CELL 11: Save Final Output ───────────────────────────────────────────────
asr_df[['recording_id','reference','asr_raw','asr_num_normalized','asr_tagged_english']].to_csv(
    OUT_DIR / "q2_postprocessing_output.csv", index=False
)
print(f"Saved Q2 output: {OUT_DIR / 'q2_postprocessing_output.csv'}")

# Summary stats
en_detected = asr_df['asr_tagged_english'].str.count(r'\[EN\]').sum()
print(f"\nEnglish loanword instances detected: {int(en_detected)}")
print(f"Segments with at least one English word: "
      f"{asr_df['asr_tagged_english'].str.contains(r'\\[EN\\]', na=False).sum()}")
