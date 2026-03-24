# =============================================================================
# Q4: Lattice-based WER Evaluation
# Platform: CPU (Kaggle or local) — pure Python + numpy
# =============================================================================
# Theory:
#   Standard WER penalizes any deviation from a single reference string.
#   But for Hindi ASR: "पाँच" vs "पांच", "14" vs "चौदह", "किताब" vs "पुस्तक"
#   are all valid transcriptions of the same audio.
#   A LATTICE replaces the flat reference with a list of BINS.
#   Each bin = one alignment position in the audio.
#   Each bin contains ALL valid lexical/spelling/numeric alternatives.
#   WER is computed against whichever alternative in each bin is closest
#   to the model prediction.
# =============================================================================

# ── CELL 1: Imports ──────────────────────────────────────────────────────────
import re, json, unicodedata
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

import os as _os
if _os.path.exists("/kaggle/working"):
    OUT_DIR = Path("/kaggle/working/output")
else:
    OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 1: ALIGNMENT UNIT CHOICE + JUSTIFICATION
# =============================================================================
print("""
ALIGNMENT UNIT: WORD
Justification:
  - Subword (BPE) units don't map cleanly to meaningful Hindi units;
    splitting "इंटरव्यू" mid-syllable obscures errors.
  - Phrase-level is too coarse — errors within a phrase go unmeasured.
  - Word-level aligns with how humans perceive ASR errors and matches
    standard WER definition. Hindi words are whitespace-delimited.
""")

# =============================================================================
# SECTION 2: VARIANT DICTIONARIES (for lattice construction)
# =============================================================================

# Spelling variants (Unicode/anusvara alternations)
SPELLING_VARIANTS: Dict[str, List[str]] = {
    'पाँच'    : ['पांच', 'पाँच'],
    'नहीं'    : ['नहीं', 'नही'],
    'हैं'     : ['हैं', 'है'],
    'वह'      : ['वह', 'वो', 'वे'],
    'किताब'   : ['किताब', 'पुस्तक', 'पोथी'],
    'मकान'    : ['मकान', 'घर', 'गृह'],
    'जल्दी'   : ['जल्दी', 'शीघ्र', 'तुरंत'],
    'खरीदी'   : ['खरीदीं', 'खरीदी'],
    'किताबें' : ['किताबें', 'किताबे', 'पुस्तकें'],
    'आँखें'   : ['आँखें', 'आंखें', 'नेत्र'],
}

# Number word ↔ digit variants
NUMBER_VARIANTS: Dict[str, List[str]] = {
    'एक'    : ['एक', '1'],
    'दो'    : ['दो', '2'],
    'तीन'   : ['तीन', '3'],
    'चार'   : ['चार', '4'],
    'पाँच'  : ['पाँच', 'पांच', '5'],
    'छह'    : ['छह', '6'],
    'सात'   : ['सात', '7'],
    'आठ'    : ['आठ', '8'],
    'नौ'    : ['नौ', '9'],
    'दस'    : ['दस', '10'],
    'ग्यारह': ['ग्यारह', '11'],
    'बारह'  : ['बारह', '12'],
    'तेरह'  : ['तेरह', '13'],
    'चौदह'  : ['चौदह', '14'],
    'पंद्रह': ['पंद्रह', '15'],
    'बीस'   : ['बीस', '20'],
    'पच्चीस': ['पच्चीस', '25'],
    'तीस'   : ['तीस', '30'],
    'पचास'  : ['पचास', '50'],
    'सौ'    : ['सौ', '100'],
    'हजार'  : ['हजार', 'हज़ार', '1000'],
    'लाख'   : ['लाख', '100000'],
    '14'    : ['14', 'चौदह'],
    '100'   : ['100', 'सौ'],
}

# Devanagari ↔ Roman script variants (English loanwords)
SCRIPT_VARIANTS: Dict[str, List[str]] = {
    'इंटरव्यू' : ['इंटरव्यू', 'interview'],
    'जॉब'      : ['जॉब', 'job'],
    'मोबाइल'   : ['मोबाइल', 'mobile'],
    'ऑफिस'     : ['ऑफिस', 'office'],
    'कंप्यूटर' : ['कंप्यूटर', 'computer'],
    'प्रोजेक्ट': ['प्रोजेक्ट', 'project'],
    'इंटरनेट'  : ['इंटरनेट', 'internet'],
    'interview' : ['interview', 'इंटरव्यू'],
    'job'       : ['job', 'जॉब'],
}

def get_all_variants(word: str) -> List[str]:
    """Return all valid alternatives for a word (spelling, number, script)."""
    variants = {word}
    for table in [SPELLING_VARIANTS, NUMBER_VARIANTS, SCRIPT_VARIANTS]:
        if word in table:
            variants.update(table[word])
        # Reverse lookup
        for k, vs in table.items():
            if word in vs:
                variants.update(vs)
    return list(variants)

# =============================================================================
# SECTION 3: EDIT DISTANCE WITH LATTICE BINS
# =============================================================================

def edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Standard Levenshtein edit distance between two word sequences."""
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=int)
    dp[:, 0] = np.arange(m+1)
    dp[0, :] = np.arange(n+1)
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return int(dp[m][n])

def word_matches_bin(word: str, bin_: List[str]) -> bool:
    """True if `word` matches any alternative in the bin (case/normalize insensitive)."""
    norm_word = unicodedata.normalize('NFC', word.strip().lower())
    for alt in bin_:
        if unicodedata.normalize('NFC', alt.strip().lower()) == norm_word:
            return True
    return False

def lattice_edit_distance(hyp_tokens: List[str], lattice: List[List[str]]) -> Tuple[int, int]:
    """
    Compute edit distance between hypothesis word sequence and a lattice.
    For each lattice bin, the cost of aligning a hypothesis word is 0
    if the word matches any alternative in that bin, else 1 (substitution).
    Returns (edit_distance, ref_length)
    """
    m = len(hyp_tokens)
    n = len(lattice)

    dp = np.zeros((m+1, n+1), dtype=int)
    dp[:, 0] = np.arange(m+1)
    dp[0, :] = np.arange(n+1)

    for i in range(1, m+1):
        for j in range(1, n+1):
            # Substitution cost: 0 if hyp token matches any alternative in bin j
            sub_cost = 0 if word_matches_bin(hyp_tokens[i-1], lattice[j-1]) else 1
            dp[i][j] = min(
                dp[i-1][j]   + 1,           # deletion
                dp[i][j-1]   + 1,           # insertion
                dp[i-1][j-1] + sub_cost,    # substitution or match
            )

    return int(dp[m][n]), n

def lattice_wer(hyp: str, lattice: List[List[str]]) -> float:
    """Compute WER for one utterance against its lattice reference."""
    hyp_tokens = hyp.strip().split()
    if not lattice:
        return 0.0
    dist, ref_len = lattice_edit_distance(hyp_tokens, lattice)
    return dist / ref_len if ref_len > 0 else 0.0

# =============================================================================
# SECTION 4: LATTICE CONSTRUCTION FROM MULTIPLE MODEL OUTPUTS
# =============================================================================

def normalize_token(w: str) -> str:
    return unicodedata.normalize('NFC', w.strip().lower())

def build_lattice(
    reference: str,
    model_outputs: List[str],
    agreement_threshold: float = 0.6,
) -> List[List[str]]:
    """
    Build a word-level lattice from the human reference + N model outputs.

    Algorithm:
    1. Align all model outputs to the reference using edit distance.
    2. For each reference position, collect all model alternatives.
    3. If >= agreement_threshold fraction of models agree on an alternative
       that differs from the reference, trust models over reference.
    4. Each bin = set of valid alternatives (reference + model alternatives).

    Args:
        reference:            Human reference transcription string
        model_outputs:        List of model prediction strings
        agreement_threshold:  Fraction of models that must agree to override reference

    Returns:
        lattice: List of bins, each bin is a list of valid word alternatives
    """
    ref_tokens = reference.strip().split()
    n_models   = len(model_outputs)

    # Align each model output to reference using Needleman-Wunsch style DP
    # Collect per-position alternatives
    position_alts: List[Dict[str, int]] = [{} for _ in ref_tokens]

    for hyp in model_outputs:
        hyp_tokens = hyp.strip().split()
        alignment = align_sequences(ref_tokens, hyp_tokens)
        for ref_idx, hyp_word in alignment:
            if ref_idx is not None and hyp_word is not None:
                w = normalize_token(hyp_word)
                position_alts[ref_idx][w] = position_alts[ref_idx].get(w, 0) + 1

    # Build lattice bins
    lattice: List[List[str]] = []
    for pos, ref_word in enumerate(ref_tokens):
        alts = position_alts[pos]
        bin_set = {normalize_token(ref_word)}

        # Add spelling/number/script variants of reference word
        for variant in get_all_variants(ref_word):
            bin_set.add(normalize_token(variant))

        # Check model agreement: if majority disagrees with reference,
        # include their alternative (models may be right, reference wrong)
        for model_word, count in alts.items():
            agreement = count / n_models
            if agreement >= agreement_threshold:
                bin_set.add(model_word)
                # Also add variants of the agreed alternative
                for variant in get_all_variants(model_word):
                    bin_set.add(normalize_token(variant))

        lattice.append(sorted(bin_set))

    return lattice

def align_sequences(ref: List[str], hyp: List[str]) -> List[Tuple[Optional[int], Optional[str]]]:
    """
    Align hypothesis to reference using edit distance traceback.
    Returns list of (ref_position, hyp_word) pairs.
    ref_position = None for insertions, hyp_word = None for deletions.
    """
    m, n = len(ref), len(hyp)
    dp = np.zeros((m+1, n+1), dtype=int)
    dp[:, 0] = np.arange(m+1)
    dp[0, :] = np.arange(n+1)

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if normalize_token(ref[i-1]) == normalize_token(hyp[j-1]) else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)

    # Traceback
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if normalize_token(ref[i-1]) == normalize_token(hyp[j-1]) else 1
            if dp[i][j] == dp[i-1][j-1] + cost:
                alignment.append((i-1, hyp[j-1]))
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append((i-1, None))  # deletion
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append((None, hyp[j-1]))  # insertion
            j -= 1

    return list(reversed(alignment))

# =============================================================================
# SECTION 5: DEMO — 5 Model Outputs + Human Reference
# =============================================================================

# Example utterance: "उसने चौदह किताबें खरीदीं"
DEMO_REFERENCE = "उसने चौदह किताबें खरीदीं"

DEMO_MODEL_OUTPUTS = [
    "उसने 14 किताबें खरीदी",       # Model 1: digit, drops anusvara
    "उसने चौदह किताबे खरीदीं",     # Model 2: drops anusvaara on किताबें
    "उसने 14 पुस्तकें खरीदीं",     # Model 3: digit + synonym
    "उसने चौदह किताबें खरीदी",     # Model 4: drops anusvara on खरीदीं
    "उसने चौदह किताबें खरीदीं",    # Model 5: exact match
]

lattice = build_lattice(DEMO_REFERENCE, DEMO_MODEL_OUTPUTS, agreement_threshold=0.6)

print("=== LATTICE CONSTRUCTION DEMO ===")
print(f"\nReference   : {DEMO_REFERENCE}")
print(f"\nModel outputs:")
for i, o in enumerate(DEMO_MODEL_OUTPUTS, 1):
    print(f"  Model {i}: {o}")

print(f"\nConstructed Lattice:")
ref_tokens = DEMO_REFERENCE.split()
for i, (ref_word, bin_) in enumerate(zip(ref_tokens, lattice)):
    print(f"  Bin {i+1} (ref='{ref_word}'): {bin_}")

# Standard WER vs Lattice WER per model
print("\n=== STANDARD WER vs LATTICE WER ===")
from jiwer import wer as std_wer_fn

print(f"{'Model':<12} {'Standard WER':>14} {'Lattice WER':>13} {'Note'}")
print("-"*65)
for i, hyp in enumerate(DEMO_MODEL_OUTPUTS, 1):
    s_wer = round(100 * std_wer_fn(DEMO_REFERENCE, hyp), 1)
    l_wer = round(100 * lattice_wer(hyp, lattice), 1)
    note  = "← unfairly penalized" if s_wer > l_wer else ("exact match" if s_wer == 0 else "")
    print(f"  Model {i}    {s_wer:>12.1f}%  {l_wer:>11.1f}%  {note}")

print("""
Interpretation:
  - Models using '14' instead of 'चौदह' get 0% lattice WER (both are valid)
  - Models dropping anusvara get 0% lattice WER (variant in bin)
  - Model 5 (exact match) has 0% both ways
  - Standard WER unfairly penalized Models 1–4 for valid transcriptions
""")

# =============================================================================
# SECTION 6: CORPUS-LEVEL LATTICE WER
# =============================================================================

def compute_corpus_lattice_wer(
    references: List[str],
    hypotheses: List[str],
    all_model_outputs: Optional[List[List[str]]] = None,
) -> float:
    """
    Corpus-level lattice WER.
    If all_model_outputs provided, builds lattice from all models.
    Otherwise builds lattice from reference only (with variants).
    """
    total_errors = 0
    total_ref_len = 0

    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if all_model_outputs:
            others = [model[i] for model in all_model_outputs]
        else:
            others = []

        lat = build_lattice(ref, others + [hyp])
        dist, ref_len = lattice_edit_distance(hyp.split(), lat)
        total_errors  += dist
        total_ref_len += ref_len

    return (total_errors / total_ref_len * 100) if total_ref_len > 0 else 0.0

# Run on multi-utterance example
references = [
    "उसने चौदह किताबें खरीदीं",
    "मुझे पाँच सौ रुपए चाहिए",
    "वह ऑफिस में इंटरव्यू दे रहा था",
]

model_outputs = {
    "Model_1": ["उसने 14 किताबें खरीदी", "मुझे 500 रुपए चाहिए", "वह office में interview दे रहा था"],
    "Model_2": ["उसने चौदह किताबे खरीदीं", "मुझे पांच सौ रुपए चाहिए", "वह ऑफिस में इंटरव्यू दे रहा था"],
    "Model_3": ["उसने 14 पुस्तकें खरीदीं", "मुझे पाँच सौ रुपये चाहिए", "वह ऑफिस में इंटरव्यू दे रहा"],
    "Model_4": ["उसने चौदह किताबें खरीदी", "मुझे 500 रुपए चाहिए", "वह ऑफिस में इंटरव्यू दे रहा था"],
    "Model_5": ["उसने चौदह किताबें खरीदीं", "मुझे पाँच सौ रुपए चाहिए", "वह ऑफिस में इंटरव्यू दे रहा था"],
}

print("=== CORPUS-LEVEL RESULTS (3 utterances, 5 models) ===")
print(f"\n{'Model':<12} {'Std WER (%)':>12} {'Lattice WER (%)':>16}")
print("-"*42)

try:
    from jiwer import wer as std_wer_fn
except ImportError:
    def std_wer_fn(r, h): return edit_distance(r.split(), h.split()) / max(1, len(r.split()))

all_hyps = list(model_outputs.values())

for model_name, hyps in model_outputs.items():
    # Standard WER (aggregate)
    total_ed = sum(edit_distance(r.split(), h.split()) for r, h in zip(references, hyps))
    total_rl = sum(len(r.split()) for r in references)
    s_wer = round(100 * total_ed / total_rl, 1)

    # Lattice WER (using all 5 models to build lattice)
    l_wer = round(compute_corpus_lattice_wer(references, hyps, all_hyps), 1)

    print(f"  {model_name:<12} {s_wer:>10.1f}%  {l_wer:>14.1f}%")

print("""
Key insight: Model_1 (uses digits/Roman script) was heavily penalized by
standard WER but gets near-0% lattice WER because the lattice bins
include both numeral and word forms, and both Devanagari/Roman script.
""")

# Save demo outputs
summary = []
for model_name, hyps in model_outputs.items():
    total_ed = sum(edit_distance(r.split(), h.split()) for r, h in zip(references, hyps))
    total_rl = sum(len(r.split()) for r in references)
    s_wer = round(100 * total_ed / total_rl, 1)
    l_wer = round(compute_corpus_lattice_wer(references, hyps, all_hyps), 1)
    summary.append({'model': model_name, 'standard_wer': s_wer, 'lattice_wer': l_wer})

pd.DataFrame(summary).to_csv(OUT_DIR / "q4_lattice_wer_results.csv", index=False)
print(f"Saved: {OUT_DIR / 'q4_lattice_wer_results.csv'}")

# =============================================================================
# SECTION 7: REAL EVALUATION — 4 Whisper Models on Actual Hindi Audio
# Run this on Kaggle (GPU) for faster inference
# Fine-tuned model (Model 5) to be added after Q1 training completes
# =============================================================================

print("\n" + "="*60)
print("REAL EVALUATION: 4 Whisper Models on Josh Talks Audio")
print("="*60)

# !pip install -q transformers torch librosa soundfile jiwer

import torch, soundfile as sf, json, urllib.request, re
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLE_RATE   = 16000
# Paths: auto-detect Kaggle vs local
import os as _os
if _os.path.exists("/kaggle/working"):
    CSV_PATH      = "/kaggle/input/datasets/sivatejareddya/josh-talks-hindi-asr/FT Data - data.csv"
    MANIFEST_PATH = "/kaggle/input/datasets/sivatejareddya/josh-talks-segments/segments_manifest.csv"
    LOCAL_SEG_DIR = Path("/kaggle/working/output/q4_samples")
else:
    CSV_PATH      = str(Path(__file__).parent / "Datasets" / "FT Data - data.csv")
    MANIFEST_PATH = str(Path(__file__).parent / "Datasets" / "segments_manifest.csv")
    LOCAL_SEG_DIR = Path(__file__).parent / "output" / "q4_samples"
LOCAL_SEG_DIR.mkdir(parents=True, exist_ok=True)
N_SAMPLES     = 20
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

_ft_path = (
    "/kaggle/input/datasets/sivatejareddya/whisper-hindi-best/whisper-hindi-best"
    if _os.path.exists("/kaggle/working")
    else str(Path(__file__).parent / "whisper-hindi-best")
)
MODELS = {
    "whisper-tiny"            : "openai/whisper-tiny",
    "whisper-base"            : "openai/whisper-base",
    "whisper-small"           : "openai/whisper-small",
    "whisper-medium"          : "openai/whisper-medium",
    "whisper-small-finetuned" : _ft_path,
}

def fix_url(url: str) -> str:
    m = re.search(r'/hq_data/hi/(\d+)/(.+)$', url)
    return f'https://storage.googleapis.com/upload_goai/{m.group(1)}/{m.group(2)}' if m else url

def get_local_audio(row: pd.Series, seg_row: pd.Series) -> Optional[str]:
    """Return local path to segment audio, downloading + slicing if needed."""
    seg_path = LOCAL_SEG_DIR / f"{seg_row['recording_id']}_{int(seg_row['segment_id']):03d}.wav"
    if seg_path.exists():
        return str(seg_path)
    # Download full recording
    wav_path = LOCAL_SEG_DIR / f"{seg_row['recording_id']}.wav"
    if not wav_path.exists():
        try:
            print(f"  Downloading recording {seg_row['recording_id']}...")
            urllib.request.urlretrieve(fix_url(row['rec_url_gcp']), wav_path)
        except Exception as e:
            print(f"  Download failed: {e}")
            return None
    # Slice segment
    try:
        trans = json.loads(urllib.request.urlopen(
            fix_url(row['transcription_url_gcp']), timeout=30).read())
        seg_data = trans[int(seg_row['segment_id'])]
        start, end = float(seg_data['start']), float(seg_data['end'])
        audio, orig_sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
        if orig_sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
        sf.write(seg_path, audio[int(start*SAMPLE_RATE):int(end*SAMPLE_RATE)], SAMPLE_RATE)
        return str(seg_path)
    except Exception as e:
        print(f"  Slice failed: {e}")
        return None

# Load manifest + source CSV for URLs
seg_df = pd.read_csv(MANIFEST_PATH).head(N_SAMPLES)
src_df = pd.read_csv(CSV_PATH).set_index('recording_id')

# Resolve local audio paths (download only missing)
audio_paths, references = [], []
for _, seg_row in seg_df.iterrows():
    src_row = src_df.loc[int(seg_row['recording_id'])]
    path = get_local_audio(src_row, seg_row)
    if path:
        audio_paths.append(path)
        references.append(seg_row['text'])

print(f"Evaluating on {len(audio_paths)} segments")

def load_audio(path: str) -> np.ndarray:
    audio, orig_sr = sf.read(path, dtype='float32', always_2d=False)
    if orig_sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
    return audio

def run_model(model_id: str, audio_paths: List[str]) -> List[str]:
    print(f"\nRunning {model_id}...")
    device = torch.device(DEVICE)
    proc  = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    results = []
    for path in audio_paths:
        try:
            audio = load_audio(path)
            feats = proc.feature_extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(device)
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
            # Filter hallucination loops (e.g. "तो तो तो तो...")
            words = pred.split()
            if len(words) > 5 and len(set(words)) < 3:
                pred = ""
            elif words and max(len(w) for w in words) > 20 and len(set(pred.replace(" ", ""))) < 6:
                pred = ""
            results.append(pred)
        except Exception as e:
            results.append("")
            print(f"  [SKIP] {path}: {e}")

    del model, proc
    torch.cuda.empty_cache()
    return results

# Run all 4 models
all_predictions = {}
for model_name, model_id in MODELS.items():
    try:
        preds = run_model(model_id, audio_paths)
        all_predictions[model_name] = preds
        print(f"  Done: {len(preds)} predictions")
    except Exception as e:
        print(f"  [FAIL] {model_name}: {e}")

# Compute Standard WER vs Lattice WER
from jiwer import wer as std_wer_fn

print("\n" + "="*65)
print(f"{'Model':<25} {'Std WER (%)':>12} {'Lattice WER (%)':>16} {'Delta':>8}")
print("-"*65)

all_hyps_list = list(all_predictions.values())
real_summary  = []

for model_name, hyps in all_predictions.items():
    # Standard WER
    s_wer = round(100 * std_wer_fn(references, hyps), 2)
    # Lattice WER
    l_wer = round(compute_corpus_lattice_wer(references, hyps, all_hyps_list), 2)
    delta = round(s_wer - l_wer, 2)
    flag  = " ← unfairly penalized" if delta > 5 else ""
    print(f"  {model_name:<23} {s_wer:>12.2f}  {l_wer:>14.2f}  {delta:>+7.2f}{flag}")
    real_summary.append({
        "model"        : model_name,
        "standard_wer" : s_wer,
        "lattice_wer"  : l_wer,
        "delta"        : delta,
    })

print("="*65)
print("(Positive delta = model was unfairly penalized by standard WER)")
print("\nNote: Add fine-tuned model after Q1 completes:")
print("  MODELS['whisper-small-finetuned'] = '/kaggle/working/whisper-hindi-best'")

# Show sample predictions from all models
print("\n--- SAMPLE PREDICTIONS (first 3 segments) ---")
for i in range(min(3, len(references))):
    print(f"\n[{i+1}] REF: {references[i]}")
    for model_name, hyps in all_predictions.items():
        print(f"     {model_name:<23}: {hyps[i]}")

# Save
result_df = pd.DataFrame(real_summary)
result_df.to_csv(OUT_DIR / "q4_real_wer_comparison.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'q4_real_wer_comparison.csv'}")
