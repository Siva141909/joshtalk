# Task Submission — AI Researcher Intern (Speech & Audio)
**Candidate:** Siva Teja Reddy A
**Company:** Josh Talks
**Date:** March 2026

---

## Dataset

**Given:** 104 Hindi audio recordings (~21.9 hours total), each paired with a JSON transcription file containing word-level timestamps and a metadata file. Hosted on GCP.

**What we built from it:** Parsed every JSON to extract segment start/end times, sliced the WAV files into sentence-level chunks, and produced a `segments_manifest.csv` with 4,929 segments. Each row has: audio path, reference text, recording ID, segment ID, duration. This manifest was the input to all four questions.

---

## Q1: Fine-tune Whisper on Hindi ASR Data

### What was given
A dataset of 104 conversational Hindi recordings with human transcriptions. The task was to fine-tune a Whisper model, evaluate it, analyze errors, and show the impact of text normalization.

### What we did
Fine-tuned `openai/whisper-small` on the 4,929 segments. Split: 80% train / 10% val / 10% test. Trained for 500 steps on Kaggle T4 GPU. Evaluated both the pretrained and fine-tuned model on the validation split using Word Error Rate (WER). Sampled 25 error utterances stratified by severity. Built an error taxonomy from real predictions. Applied text normalization and measured its impact on WER.

### How we did it

**Training config:**

| Parameter | Value |
|-----------|-------|
| Base model | openai/whisper-small (244M params) |
| Steps | 500 |
| Batch size | 16 per device |
| Gradient accumulation | 2 steps |
| Learning rate | 1e-5 |
| Warmup steps | 50 |
| Mixed precision | fp16 (T4 GPU) |
| Metric | WER (lower is better) |

**Evaluation:** Used `WhisperProcessor` + `WhisperForConditionalGeneration` with direct inference (`model.generate()`) instead of the pipeline API, to avoid hallucination issues caused by the pipeline's internal post-processing. Applied `temperature=0.0` and `condition_on_prev_tokens=False` to suppress repetition loops.

**Error sampling:** Stratified by WER severity — 10 severe (WER > 50%), 10 moderate (20–50%), 5 mild (< 20%). No cherry-picking.

### Results

| Model | Raw WER | Normalized WER |
|-------|---------|----------------|
| Whisper-small (pretrained) | 255.93% | 255.59% |
| Whisper-small (fine-tuned) | 36.21% | **33.85%** |

**86% relative WER reduction** from fine-tuning. Normalization added a further 2.36 point improvement on the fine-tuned model.

**25 sampled error utterances (representative):**

| # | WER | Reference | Prediction |
|---|-----|-----------|------------|
| 01 | 52.8% | उससे अपना पढ़ने की चाह बढ़ती थी... | दूससे अपना पड़ने की चाह पड़ती थी... |
| 04 | 60.0% | सबसे बड़ा त्यौहार है जी | चपते बड़ा त्यावा रही है जी |
| 08 | 76.5% | इमेज़न करने लगता था कि ये दुनिया है... | इमजन करने लगता रहा कि यह दुनियां है... |
| 12 | 23.8% | हमारे यहां तो सबसे बड़ी रश्म तो शादी होती है... | हमारे यहां तो जबसे बड़ी रसम तो शादी होती है... |
| 22 | 9.1% | तो मतलब वो दिन मेरे जीवन का सबसे बड़ा दिन होगा | तो मतलब वो दिन मेरे जीवन का सबसे बड़ा तिन होगा |

**Error taxonomy (from actual predictions):**

| Category | Description | Example |
|----------|-------------|---------|
| CAT-1: Phonetic substitution | Acoustically similar word swapped | `रश्म` → `रसम`, `त्यौहार` → `त्यावा` |
| CAT-2: Unicode/spelling variant | Valid alternate form counted as error | `हां` → `हाँ`, `नहीं` → `नही` |
| CAT-3: Backchannel confusion | Single-word fillers misheard | `हां` → `हाँ` (100% WER on 1-word segment) |
| CAT-4: Word boundary error | Compound split/merged wrong | `बतादो` → `बता दो`, `ऑंलाइन` → `ऑनलाइन` |
| CAT-5: Partial truncation | Transcription cuts off mid-sentence | Last 2–3 words dropped |

**Text normalization impact:**

| Before normalization | After normalization | Why |
|----------------------|---------------------|-----|
| `हाँ` vs `हां` → 100% WER | → 0% WER | Same word, different Unicode anusvara |
| `हां.` vs `हां` → 100% WER | → 0% WER | Trailing punctuation removed |
| `हा बिल्कुल, बिल्कुल ,जी.` vs `हां बिल्कुल बिल्कुल जी` → 75% WER | → 25% WER | Punctuation + anusvara normalization |

### Why these results

**Why pretrained WER is 255%:** Whisper-small pretrained on multilingual data had no domain exposure to conversational Hindi. It hallucinated in English, repeated phrases, and confused similar-sounding words. WER > 100% means the model inserted more words than the reference contained.

**Why fine-tuned WER is 36%:** 500 steps of in-domain training gave the model exposure to the conversational Hindi vocabulary, speaking style, and background conditions in Josh Talks recordings. The model learned to produce plausible Hindi output rather than hallucinate.

**Why normalization helps:** 6 of the 25 sampled errors had WER purely from Unicode variant differences (`हां`/`हाँ`) or punctuation — things that are not acoustic errors at all. Normalization removes this noise from the metric.

**Why WER is still 36% and not lower:** The dataset is conversational Hindi with heavy code-switching, regional accents, and background noise. 500 training steps on ~4,900 segments is relatively limited. More steps, data augmentation, and a larger base model (whisper-medium) would reduce it further.

---

## Q2: Post-processing Pipeline — Number Normalization + English Word Detection

### What was given
Raw ASR output from Whisper on Hindi audio. The task was to build a pipeline that: (1) converts Hindi number words to digits, and (2) detects English loanwords transcribed in Devanagari.

### What we did
Built a two-stage rule-based pipeline. Stage 1 normalizes number words to digits with idiom protection. Stage 2 detects English loanwords using a whitelist and phonetic pattern heuristics. Applied it to Whisper-small (pretrained) output on 20 segments.

### How we did it

**Number normalization:**
- Built a lookup table of all Hindi number words from 0–99 plus multipliers (`सौ`=100, `हजार`=1000, `लाख`=100000, `करोड़`=10000000)
- Greedy longest-match scan: tries to convert the longest possible consecutive sequence of number words into a single digit
- Protected idioms that contain number words but must NOT be converted: `दो-चार` (means "a few"), `नौ दो ग्यारह` (means "to flee"), `एक-आध` (means "once or twice"), `सात समंदर` (poetic phrase)

```
Input:  "उसने चौदह किताबें खरीदीं"
Output: "उसने 14 किताबें खरीदीं"

Input:  "दो-चार बातें हुईं"
Output: "दो-चार बातें हुईं"   ← idiom preserved
```

**English loanword detection:**
- Whitelist of 40+ English words correctly transcribed in Devanagari per Josh Talks guidelines: `इंटरव्यू`, `जॉब`, `ऑफिस`, `मीटिंग`, `प्रोजेक्ट`, `कॉलेज`, `ट्रेनिंग` etc.
- Phonetic heuristics for unlisted loanwords: words starting with `ऑ` (open-O, non-native Hindi vowel), ending in `-शन` (-tion), `-टिंग` (-ting), `-मेंट` (-ment), or starting with `वेब-`/`ऐप-`
- Tagged output format: `[EN]इंटरव्यू[/EN]`

**Pipeline demonstration on known examples:**
```
Input:  "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई"
Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई"

Input:  "उसने ऑनलाइन कोर्स किया और बहुत कुछ सीखा"
Output: "उसने [EN]ऑनलाइन[/EN] [EN]कोर्स[/EN] किया और बहुत कुछ सीखा"
```

### Results

| Metric | Value |
|--------|-------|
| Total segments processed | 20 |
| Numbers normalized | 0 |
| English loanwords detected | 0 |

### Why these results

The 20 evaluated segments are all from recording 825780 — a field research conversation about tribal communities (`खांड जनजाति`) in a jungle area. This domain has no number words (no prices, dates, quantities) and no English loanwords (no tech, career, or modern vocabulary). The pipeline did not trigger because the trigger conditions were absent, not because the pipeline is broken.

The pipeline correctness is demonstrated above on representative sentences. On general Josh Talks content (career advice, interviews, education topics) which contains sentences like "मेरा इंटरव्यू हुआ" or "तीन साल का कोर्स है", both stages would activate correctly.

---

## Q3: Hindi Spelling Error Detection on ~7,354 Unique Words

### What was given
A requirement to classify ~1,77,000 unique words (or available words from our data) as correctly or incorrectly spelled, using only free resources, on CPU.

### What we did
Extracted all unique word tokens from our 4,929-segment transcription dataset — yielding 7,354 unique words. Built a multi-signal classifier using a downloaded Hindi frequency vocabulary as the reference, plus rule-based misspelling pattern detection.

### How we did it

**Reference vocabulary:** Downloaded `hi_50k.txt` from `hermitdave/FrequencyWords` — a 50,000-word Hindi frequency list. This served as the ground-truth vocabulary.

**7-signal classifier:**

| Signal | Direction | Description |
|--------|-----------|-------------|
| In reference vocabulary | +2 correct | Word found in 50k Hindi frequency list |
| Known Devanagari loanword | High confidence correct | 35-word whitelist of English words per Josh Talks guidelines |
| Roman script present | High confidence incorrect | ASCII letters in a Devanagari transcription |
| Misspelling patterns | −1 per match | Triple char repetition, halant at end, matra at start, 3+ consecutive vowels, double anusvara, visarga mid-word |
| Not NFC normalized | −1 | Unicode composition mismatch |
| Single character (non-numeral) | −1 | Likely transcription noise |
| Word length > 20 chars | −1 | Unusual in standard Hindi |

**Decision:** net score ≥ 2 → correct/high confidence; = 1 → correct/medium; = 0 → correct/low; ≤ −1 → incorrect.

### Results

| Label | Count | % |
|-------|-------|---|
| Correctly spelled | 7,259 | 98.7% |
| Incorrectly spelled | 95 | 1.3% |

Confidence breakdown for correct: High ~6,800 | Medium ~300 | Low ~159
Confidence breakdown for incorrect: High ~60 | Medium ~25 | Low ~10

### Why these results

**Why 98.7% correct:** The transcription data was human-annotated and the source audio is from real conversations — people generally speak standard Hindi. Most words appear in the reference vocabulary.

**Why 95 flagged as incorrect:** These fall into: Roman-script tokens that slipped through (e.g. `ok`, `job`), very short single-character noise tokens, and words with abnormal Unicode patterns (e.g. non-NFC normalized characters from transcription tools).

**Where the classifier is unreliable (low confidence bucket):**
- **Proper nouns** — names, cities, brand names (`नरेंद्र`, `मुंबई`, `जियो`) are not in standard frequency lists → classified as OOV → low confidence. These are correct but our classifier cannot confirm them.
- **English loanwords in Devanagari** — words like `प्रेजेंटेशन`, `डेडलाइन` are correct per Josh Talks transcription guidelines but absent from Hindi dictionaries. Fix: expand the whitelist or use a transliteration model.

---

## Q4: Lattice-based WER Evaluation

### What was given
The task to design a lattice-based WER metric — an improvement over standard WER that handles valid alternative transcriptions — and evaluate 5 models using it.

### What we did
Implemented a word-level lattice WER from scratch. Built a lattice from multiple model outputs + human reference. Compared standard WER vs lattice WER across 5 Whisper models on 20 real Josh Talks audio segments.

### How we did it

**Why standard WER is unfair for Hindi:**
Standard WER treats `"चौदह"` and `"14"` as completely different words — 100% substitution cost. But both are valid transcriptions of the same spoken word. Same for `"पाँच"`/`"पांच"` (Unicode variants), `"किताब"`/`"पुस्तक"` (synonyms), `"इंटरव्यू"`/`"interview"` (script variants). A model should not be penalized for valid alternatives.

**Alignment unit: Word**
Word-level was chosen over subword (BPE units don't map cleanly to meaningful Hindi units — splitting `"इंटरव्यू"` mid-syllable obscures the error) and over phrase-level (too coarse — intra-phrase errors go unmeasured).

**Lattice construction:**
Each reference word position becomes a bin. The bin starts with the reference word, then expands by:
1. Pre-built variant tables: spelling variants (`पाँच`↔`पांच`), number variants (`चौदह`↔`14`), script variants (`इंटरव्यू`↔`interview`)
2. Model agreement: if ≥60% of the 5 models output the same alternative at that position, it is added to the bin as a valid transcription

**Lattice WER formula:**
```
lattice_edit_distance = Levenshtein distance where substitution cost = 0
                        if hypothesis word matches ANY word in that bin

lattice_WER = lattice_edit_distance(hypothesis, lattice) / len(lattice)
```

**Synthetic demo (utterance: "उसने चौदह किताबें खरीदीं"):**

| Model output | Standard WER | Lattice WER | Why different |
|-------------|-------------|-------------|---------------|
| "उसने 14 किताबें खरीदी" | 40% | 0% | `14` is in bin for `चौदह`; `खरीदी` is variant of `खरीदीं` |
| "उसने चौदह किताबें खरीदीं" | 0% | 0% | Exact match |

### Results — 5 Model Comparison on Real Audio (20 segments)

| Model | Standard WER | Lattice WER | Delta |
|-------|-------------|-------------|-------|
| whisper-tiny | 126.63% | 126.63% | +0.00% |
| whisper-base | 110.40% | 110.40% | +0.00% |
| whisper-small (pretrained) | 89.32% | 89.04% | +0.28% |
| whisper-medium | 75.17% | 74.90% | +0.27% |
| **whisper-small (fine-tuned)** | **33.70%** | **33.43%** | **+0.27%** |

**Sample predictions — Segment 3 ("जंगल का सफर होता है..."):**

| Model | Prediction |
|-------|------------|
| whisper-tiny | `Jandol kasafar... ufffff...` (Roman hallucination) |
| whisper-base | `جنگل کا سفر...` (Urdu script output) |
| whisper-small | `जंगल का सबवर भड़ा है जब लगने की लिए गयतना...` |
| whisper-medium | `जब हम रहने की लिए गयते न, तो चाहती के साथ...` |
| **fine-tuned** | `जंगल का सफर होता है जब हम रहने के लिए गए थे ना तो चाहते के साथ...` |

### Why these results

**Why fine-tuned is 33.70% vs medium's 75.17%:** The fine-tuned model was trained on this exact domain (conversational Hindi from Josh Talks). Whisper-medium is a larger model but has never seen this domain. Domain-specific fine-tuning outweighs model size.

**Why tiny/base WER is >100%:** These models had no Hindi exposure adequate for this domain. Tiny produced Roman hallucinations, base produced Urdu-script output. Both inserted more words than the reference, pushing WER above 100%.

**Why the lattice delta is small (+0.27–0.28%):** The 20 test segments are from a jungle fieldwork conversation — no number words, no English loanwords, no script mixing. The lattice bins had almost no opportunities to provide alternatives over what standard WER already accepted. On general conversational data with numbers and loanwords, the delta would be significantly larger (demonstrated in the synthetic demo where it went from 40% → 0%).

**Why tiny/base have 0% delta:** Their outputs were complete hallucinations — Roman text, Urdu script, character loops. The lattice bins contain valid Hindi alternatives, not English letters or Urdu characters. So no bin could give a zero-cost match; lattice and standard WER are identical.

---

## Output Files

| File | Description |
|------|-------------|
| `output/q1_wer_results.csv` | Pretrained vs fine-tuned WER (raw and normalized) |
| `output/q1_sampled_errors.csv` | 25 stratified error utterances with reference, prediction, WER |
| `output/q2_postprocessing_output.csv` | 20 segments: raw ASR, normalized text, detected loanwords |
| `output/q3_spelling_classification.csv` | 7,354 words with label, confidence level, and reason |
| `output/q4_real_wer_comparison.csv` | 5-model standard WER vs lattice WER comparison |

## Repository

GitHub: https://github.com/Siva141909/joshtalk
