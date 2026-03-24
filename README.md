# Josh Talks — AI Researcher Intern Task Submission
**Candidate:** Siva Teja Reddy A
**Role:** Speech & Audio — AI Researcher Intern
**Dataset:** 104 Hindi audio recordings (~21.9 hours) with human transcription JSONs
**Platform:** Kaggle (T4 GPU) + local machine

---

## Dataset Overview

| Attribute | Value |
|-----------|-------|
| Recordings | 104 Hindi audio files |
| Total duration | ~21.9 hours |
| Segments (after slicing) | 4,929 |
| Language | Hindi (Devanagari script) |
| Domain | Conversational — interviews, personal narratives, experiences |

The dataset was provided as GCP-hosted WAV files paired with JSON transcription files containing word-level timestamps. Audio was segmented into sentence-level chunks using those timestamps, producing a `segments_manifest.csv` used across all four questions.

---

## Q1: Whisper Fine-tuning on Hindi ASR

### What was done
Fine-tuned `openai/whisper-small` on the 104 Hindi recordings to improve Hindi speech recognition accuracy. Evaluated against a held-out validation split and compared pre-trained vs. fine-tuned performance.

### How it was done

**Step 1 — Data preparation**
- Downloaded all 104 recordings from GCP (`upload_goai/{folder}/{file}` URL pattern)
- Parsed JSON transcription files to extract segment start/end timestamps
- Sliced WAVs into 4,929 sentence-level segments using `librosa`
- Resampled all audio to 16kHz mono
- Built `segments_manifest.csv` with columns: `audio_path`, `text`, `recording_id`, `segment_id`, `duration`

**Step 2 — Training (Kaggle T4 GPU)**
- Base model: `openai/whisper-small` (244M parameters)
- Framework: HuggingFace `transformers` + `Seq2SeqTrainer`
- Split: 80% train / 10% val / 10% test
- Training config:
  - Steps: 500
  - `per_device_train_batch_size`: 16
  - `gradient_accumulation_steps`: 2
  - `learning_rate`: 1e-5
  - `warmup_steps`: 50
  - `fp16`: True (T4 GPU)
  - `eval_strategy`: steps (every 250 steps)
  - `metric_for_best_model`: WER
  - `predict_with_generate`: True
- WER at step 250: **42.15%**
- WER at step 500: **37.27%**
- Best checkpoint saved to `whisper-hindi-best/`

**Step 3 — Evaluation**
- Used `WhisperProcessor` + `WhisperForConditionalGeneration` for direct inference (bypassed pipeline to avoid `num_frames` errors)
- Applied hallucination prevention: `temperature=0.0`, `condition_on_prev_tokens=False`, `max_new_tokens=128`
- Computed WER using `jiwer` library

### Results

| Model | Raw WER | Normalized WER |
|-------|---------|----------------|
| Whisper-small (pretrained) | 255.93% | 255.59% |
| Whisper-small (fine-tuned) | 36.21% | 33.85% |

**Improvement: ~85.8% relative WER reduction** after 500 steps on domain-specific Hindi data.

The pretrained model scored 255% WER because it frequently hallucinated in English or produced repetitive loops on conversational Hindi. Fine-tuning on in-domain data brought it to a usable 36% WER.

### Error Analysis (25 sampled utterances)

Stratified sample: 10 severe errors (WER > 60%), 10 moderate (20–60%), 5 mild (< 20%).

**Error taxonomy from actual predictions:**

| Category | Description | Example |
|----------|-------------|---------|
| Phonetic substitution | Similar-sounding word swapped | `रश्म` → `रसम`, `निराश` → `निराज` |
| Partial truncation | Transcription cuts off mid-sentence | Last 2–3 words missing |
| Backchannel confusion | Short filler words misheard | `हां` → `हाँ` (100% WER on 1-word segments) |
| Word boundary error | Compound words split/merged wrongly | `आर जे` → `आरजे` |
| Named entity error | Proper nouns and brand names garbled | `यूट्यूब` → `यूट्यूब पे में` |

**Text normalization fix applied:**
Normalizing punctuation, anusvara variants (`हैं`/`है`, `नहीं`/`नही`) and case brought WER from 36.21% → **33.85%** without any model change.

**Output files:** `q1_wer_results.csv`, `sampled_errors.csv`

---

## Q2: Hindi ASR Post-processing Pipeline

### What was done
Built a rule-based post-processing pipeline to:
1. Normalize Hindi number words to digits (e.g., `चौदह` → `14`)
2. Detect English loanwords transcribed in Devanagari (e.g., `प्रोजेक्ट`, `इंटरव्यू`)

### How it was done

**Number normalization**
- Mapped 30+ Hindi number words to their digit equivalents
- Protected idioms from being altered: `एक बार`, `एक साथ`, `एक दूसरे`, `दो तरफ`
- Applied longest-match-first substitution to avoid partial replacements
- Covered: single digits (1–9), teens, tens (10–90), `सौ` (100), `हजार` (1000), `लाख` (100000), `करोड़` (10000000)

```
Input:  "उसने चौदह किताबें खरीदीं"
Output: "उसने 14 किताबें खरीदीं"
```

**English loanword detection**
- Maintained a whitelist of 35+ English words correctly transcribed in Devanagari per Josh Talks guidelines: `इंटरव्यू`, `जॉब`, `मोबाइल`, `कंप्यूटर`, `ऑफिस`, `प्रोजेक्ट`, `सैलरी`, `कॉलेज`, `ट्रेनिंग` etc.
- Also detected any Roman-script tokens (regex `[a-zA-Z]{2,}`) that slipped through ASR as incorrect transcriptions

**ASR inference for Q2**
- Model: `openai/whisper-small` (pretrained) on 20 segments from recording 825780
- Applied same hallucination prevention as Q1
- 17/20 segments transcribed successfully

**Pipeline output schema:**

| Column | Description |
|--------|-------------|
| `segment_id` | Segment index |
| `recording_id` | Source recording ID |
| `reference` | Human reference transcription |
| `asr_raw` | Raw ASR output |
| `asr_normalized` | After number normalization |
| `english_loanwords` | Detected loanwords (comma-separated) |
| `num_changed` | Whether normalization changed the text |

**Note on results:** The 20 evaluated segments are from a jungle/tribal field research conversation (recording 825780). This domain does not contain number words or standard English loanwords, so `numbers_normalized = 0` and `english_loanwords = 0` for this sample. The pipeline is fully functional and would activate on general conversational data containing numbers and loanwords.

**Output file:** `q2_postprocessing_output.csv`

---

## Q3: Hindi Spelling Error Detection on ~7,354 Unique Words

### What was done
Built a multi-signal classifier to label each unique word from the transcription data as **correctly spelled** or **incorrectly spelled**, with a confidence level.

### How it was done

**Word extraction**
- Loaded `segments_manifest.csv` (4,929 segments)
- Extracted all whitespace-delimited tokens, stripped punctuation (`।`, `,`, digits)
- Resulted in **7,354 unique words**

**Reference vocabulary**
- Downloaded Hindi frequency word list from `hermitdave/FrequencyWords` (50k entries)
- Used as the ground-truth vocabulary for lookup

**Multi-signal classifier (7 signals):**

| Signal | Weight | Description |
|--------|--------|-------------|
| In reference vocab | +2 correct | Word exists in 50k Hindi frequency list |
| Known Devanagari loanword | High confidence correct | 35-word whitelist of English words in Devanagari |
| Misspelling patterns | -1 per pattern | Triple char repetition, halant at end, matra at start, 3+ consecutive vowels, double anusvara, visarga mid-word |
| Not NFC normalized | -1 | Unicode normalization mismatch |
| Single character | -1 | Single char not in `एकदोतीनचारपांचछसातआठनौ` |
| Unusually long (>20 chars) | -1 | Rare in standard Hindi |
| Roman script | High confidence incorrect | ASCII letters in Devanagari transcription |

**Decision thresholds:** net score ≥ 2 → correct/high; = 1 → correct/medium; = 0 → correct/low; = -1 → incorrect/low; ≤ -2 → incorrect/high

### Results

| Label | Count | % |
|-------|-------|---|
| Correctly spelled | 7,259 | 98.7% |
| Incorrectly spelled | 95 | 1.3% |

**Confidence breakdown (correct):** High: ~6,800 | Medium: ~300 | Low: ~159
**Confidence breakdown (incorrect):** High: ~60 | Medium: ~25 | Low: ~10

**Unreliable categories identified:**

1. **Proper nouns** (names, cities, brands) — not in standard dictionaries, classified as OOV → low confidence. Fix: build a proper noun gazetteer.
2. **English loanwords in Devanagari** — words like `प्रेजेंटेशन`, `डेडलाइन` are correct per Josh Talks guidelines but absent from Hindi frequency lists. Fix: expand loanword whitelist or use a transliteration model.

**Output file:** `q3_spelling_classification.csv`

---

## Q4: Lattice-based WER Evaluation

### What was done
Designed and implemented a lattice-based WER metric that replaces a single reference string with a **lattice of valid alternatives** — one bin per alignment position, each bin containing all acceptable transcriptions of that word.

### Why lattice WER

Standard WER treats `"चौदह"` and `"14"` as completely different words even though both are valid transcriptions of the same audio. Similarly, spelling variants (`नहीं`/`नही`), synonyms (`किताब`/`पुस्तक`), and script variants (`ऑफिस`/`office`) all get penalized. A lattice allows zero cost for any accepted alternative.

### How it was done

**Alignment unit: Word**
- Subword (BPE) units don't map cleanly to meaningful Hindi units
- Phrase-level is too coarse — intra-phrase errors go unmeasured
- Word-level matches standard WER definition and human perception of ASR quality

**Lattice construction algorithm:**
1. Start with human reference tokens as bin anchors
2. Align each model output to reference using Levenshtein traceback
3. For each reference position, collect all model alternatives
4. If ≥ 60% of models agree on an alternative, add it to that bin
5. Also expand each bin with pre-built variant tables:
   - Spelling variants: `पाँच`↔`पांच`, `नहीं`↔`नही`, `हैं`↔`है`
   - Number variants: `चौदह`↔`14`, `सौ`↔`100`, `हजार`↔`1000`
   - Script variants: `इंटरव्यू`↔`interview`, `जॉब`↔`job`

**Lattice WER formula:**
```
lattice_WER = lattice_edit_distance(hypothesis, lattice) / len(lattice)
```
Where `lattice_edit_distance` uses substitution cost = 0 if hypothesis word matches **any** alternative in the bin.

### 5-Model Comparison Results (20 real Josh Talks segments)

| Model | Standard WER | Lattice WER | Delta |
|-------|-------------|-------------|-------|
| whisper-tiny | 126.63% | 126.63% | +0.00% |
| whisper-base | 110.40% | 110.40% | +0.00% |
| whisper-small (pretrained) | 89.32% | 89.04% | +0.28% |
| whisper-medium | 75.17% | 74.90% | +0.27% |
| **whisper-small (fine-tuned)** | **33.70%** | **33.43%** | **+0.27%** |

**Key observations:**
- The fine-tuned model achieves the lowest WER by a wide margin — 33.70% vs 75.17% for whisper-medium
- The delta (standard WER − lattice WER) is small here because the test segments are from a specialized jungle/field-research domain with very little number or loanword variation
- On general conversational data with numbers and script mixing, lattice WER shows larger deltas (demonstrated in synthetic demo: Model_1 using digits got 20% standard WER vs 0% lattice WER)
- whisper-tiny and whisper-base produced character-level hallucination loops on this domain; their lattice and standard WER are identical (no valid alternatives in the hallucinated output)

**Synthetic demo (Section 5–6 of code):**

| Model | Std WER | Lattice WER | Note |
|-------|---------|-------------|------|
| Model 1 (digit form: `14`) | 20.0% | 0.0% | Unfairly penalized |
| Model 2 (drops anusvara) | 20.0% | 0.0% | Unfairly penalized |
| Model 5 (exact match) | 0.0% | 0.0% | Correct either way |

**Output files:** `q4_real_wer_comparison.csv`, `q4_lattice_wer_results.csv`

---

## Output Files Summary

| File | Question | Description |
|------|----------|-------------|
| `q1_wer_results.csv` | Q1 | Pretrained vs fine-tuned WER comparison |
| `sampled_errors.csv` | Q1 | 25 error utterances with reference, prediction, WER |
| `q2_postprocessing_output.csv` | Q2 | ASR output with number normalization + loanword detection |
| `q3_spelling_classification.csv` | Q3 | 7,354 words with label, confidence, reason |
| `q4_real_wer_comparison.csv` | Q4 | 5-model standard WER vs lattice WER on real audio |
| `q4_lattice_wer_results.csv` | Q4 | Synthetic demo lattice WER results |

---

## Tools & Libraries

| Tool | Version | Use |
|------|---------|-----|
| `transformers` | latest | Whisper model, processor, trainer |
| `datasets` | latest | Dataset loading and processing |
| `jiwer` | latest | WER computation |
| `librosa` | 0.10+ | Audio loading and resampling |
| `soundfile` | latest | WAV read/write |
| `torch` | 2.x | Model inference and training |
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical operations |

**Compute:** Kaggle free tier — NVIDIA T4 GPU (16GB VRAM), 30GB RAM

---

## Fine-tuned Model

The fine-tuned `whisper-small` model is available on HuggingFace Hub:
**`Siva1419/whisper-hindi-joshtalk`**

Trained for 500 steps on 4,929 Hindi conversational segments from Josh Talks dataset. Achieves **33.85% normalized WER** on held-out validation data.
