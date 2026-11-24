Experiment plan · TranslateGemma-12B · Macedonian ↔ Albanian · final, 23 November 2026

# Does learning a term's definition help a model translate it?

Службен весник ↔ Gazeta Zyrtare + VERBIS

TranslateGemma-12B is trained in three steps, full-weight on two A100 80GB cards. TG-10 learns direct Macedonian–Albanian from about a third of the deduplicated OPUS set and checks on the way whether that much helps, TG-20 learns legal language from the Gazette, and both are saved as endpoints. The TG-30 arms then branch from TG-20 to test one idea: does a model taught legal terms with their definitions translate better than one taught the terms alone, or one given the same GPU time of plain parallel text?

36%

of the deduplicated OPUS set trains TG-10 (1,523,490 rows)

68 / 16 / 16

Gazette training split: TG-20, arms, fresh text

9

core comparison runs: 3 arms × 3 seeds

3.5 days

core GPU time; 5.0 with every optional run

[Plan](#plan)[Questions](#questions)[Example](#idea)[NLLB pilot](#pilot)[Setup](#setup)[Data](#data)[Order of work](#order)[Experiments](#experiments)[Fairness](#fair)[Scoring](#scoring)[Open items](#open)

01 · THE PLAN AT A GLANCE

## Train the shared steps once, then branch

Read left to right. Each box is one experiment; its tag is the name used in both CSV files. TG-10 and TG-20 are trained once, saved, and reused by everything to their right.

Trained once, saved Central comparison: TG-32 vs TG-33 Core run Optional run

Trained once

General adaptation (TG-10) and legal adaptation (TG-20) each run a single time and are saved. Every arm loads TG-20's weights, so the nine core arm runs repeat none of that work.

Kept apart on purpose

TG-20 trains on 68% of the Gazette training issues. The rest stays unseen, so the arms learn from new legal text and the replay arm gets parallel sentences no model has met.

02 · RESEARCH QUESTIONS

## Which comparison answers which question

Each question is settled by comparing runs that differ in one thing. RQ3 and RQ4 need no extra training: they're read from the same TG-32 and TG-33 translations.

| Question | In plain words | Decided by | Measured on |
| --- | --- | --- | --- |
| Central (a) | Do definitions help the model use trained terms correctly in new sentences? | TG-33vsTG-32  TG-35 checks why | Term success on reference-confirmed legal terms |
| Central (b) | Do definitions make translation better overall? | TG-33vsTG-32  both also vs TG-31 | chrF++ and BLEU on Gazette test, FLORES+ and NTREX-128 |
| RQ1 | Is dictionary training worth more than the same GPU time spent on plain parallel sentences? | TG-33vsTG-31  TG-32vsTG-31 | Term success and legal chrF++ |
| RQ2 | Does general quality survive each step, and what protects it? | TG-10vsTG-00  TG-20vsTG-10  TG-21vsTG-20  TG-34vsTG-31 | FLORES+ and NTREX-128 chrF++; arms may fall at most 1 point below TG-20 |
| RQ3 | Which terms gain the most from definitions? | TG-33minusTG-32  split by term group | Term success by frequency, word count and spelling similarity |
| RQ4 | Does the effect hold outside law? | TG-33vsTG-32  non-legal entries only | Term success per domain: finance, the fourth VERBIS domain, general, media literacy |
| RQ5 | Do general word and phrase pairs, with no definitions, help general translation? | TG-36vsTG-32  optional arm | FLORES+ and NTREX-128 chrF++; term success on general words |

03 · THE IDEA IN ONE EXAMPLE

## One term, three ways to spend the same GPU time

Macedonian *застареност* means statute of limitations; the Albanian legal term is *parashkrim*. The two words share nothing, so the model can't guess one from the other. (The pair is illustrative and hasn't been checked against VERBIS.)

| Arm | On top of the shared stream, the model sees | It must output |
| --- | --- | --- |
| TG-31 | New Gazette sentence pairs, as many tokens as TG-33's word task | Their translations |
| TG-32 | застареност | parashkrim |
| TG-33 | застареност plus its Macedonian VERBIS definition | parashkrim |
| Test, every model | Барањето е одбиено поради застареност. | Kërkesa është refuzuar për shkak të parashkrimit. |

At test time no model gets a dictionary or a hint. The correct output uses the inflected form *parashkrimit*, which the automatic term check accepts because it matches the opening characters *parashkr*.

04 · WHY THIS DESIGN

## What the NLLB pilot already showed

A first round on NLLB-200-distilled-600M compared the base model (E00), Gazette plus VERBIS terms and definitions (E01), and Gazette alone (E02), one seed each.

| BLEU, MK→SQ | E00 base | E01 Gazette + VERBIS | E02 Gazette only |
| --- | --- | --- | --- |
| Gazette test | 27.74 | 54.02 | 54.28 |
| FLORES+ devtest | 22.08 | **19.45** | 15.99 |
| NTREX-128 | 23.14 | **18.13** | 15.25 |

SQ→MK shows the same pattern: E01 stays 2.1 BLEU ahead of E02 on both general sets.

- **VERBIS protected general quality.** Legal BLEU roughly doubled either way, but the run with VERBIS kept 2.1 to 3.5 BLEU more of its general quality.
- **The pilot can't say why.** E01 added term pairs and definition translation at once, so either could be responsible, or simply having non-Gazette text in the mix.
- **So this design pulls the parts apart.** TG-31 tests more parallel legal text, TG-32 term pairs, TG-33 definitions as context, and TG-34 definition translation on its own. General replay in TG-20 and the arms guards against the forgetting the pilot showed.

05 · TRAINING SETUP

## Full-weight training, TranslateGemma's own recipe

Every training run uses the same configuration. Only the data and the starting checkpoint change.

| **Hardware** | 2× A100 80GB |
| --- | --- |
| **What trains** | All weights except the token embeddings (the output layer shares them) and the vision encoder, as in TranslateGemma's SFT: about 10.8B of 12.2B parameters |
| **Optimizer** | Adafactor without momentum, fp32 master weights, bf16 compute |
| **Memory** | ≈90 GB for weights, gradients and optimizer, ≈45 GB per GPU when sharded, plus activations |
| **Sharding** | FSDP2, which keeps 2-D weight shapes so Adafactor stays factored; gradient checkpointing; fused cross-entropy |
| **Batches** | Examples up to 2,048 tokens, packed; ≈64k tokens per optimizer step: micro-batches of 8k tokens per GPU, 4 accumulation steps |
| **Prompt and loss** | TranslateGemma's native translation prompt (≈80 tokens) for every sentence pair; a short separate instruction for word tasks; loss on target tokens only |
| **Learning rate** | 1e-5, fixed for every run, below TranslateGemma's own 1e-4 because training continues from an already tuned model. TG-10: warm-up over 3% of steps, then constant, then a (1−sqrt) cooldown over the final ≈15%. Every other run: warm-up over 3%, cosine decay to 10% |
| **Speed assumed** | ≈2,500 tokens per second, 9M an hour. TG-10's training log gives the real figure; every hour on this page scales with it |
| **Saved** | TG-10 and TG-20 as endpoints (bf16 weights ≈24 GB, plus resumable state until the next step is checked); TG-10's constant-rate checkpoints at ≈0.6M, ≈1.0M and ≈1.48M examples with optimizer state, kept until the size check is settled; every other run as bf16 weights; ≈0.7 TB in all |

06 · DATA

## What each step trains on

Runs train on named dataset tags, never on raw corpora. TG-D0 builds and freezes every tag before GPU work starts, and `tg_datasets.csv` lists each one source by source. Token counts assume typical segment lengths; `count_tokens.py` replaces them with measured ones.

### General step: 35.7% of the deduplicated OPUS set

| Source | Rows in TG-10 | Directions | Rule | Share of TG-10 examples | Tokens |
| --- | --- | --- | --- | --- | --- |
| **SETIMES** v2 human translation, Balkan news | 197,316 | both | all rows | 22.6% | 55.2M |
| **TED2020** v1 human translation of TED talks | 15,970 | both | all rows | 1.8% | 3.8M |
| **ELRC-5067-SciPar** v1 new scientific abstracts | 3,704 | both | all rows | 0.4% | 1.0M |
| **GlobalVoices** v2018q4 human translation, citizen media | 1,737 | both | all rows | 0.2% | 0.5M |
| **EUbookshop** v2 new EU publications | 1,674 | both | all rows | 0.2% | 0.5M |
| **NeuLab-TedTalks** v1 new TED talk transcripts | 1,130 | both | all rows | 0.1% | 0.3M |
| **Tatoeba, ELRC-wikipedia_health**  short sentences | 13 | both | all rows | 0.0% | 0.0M |
| **OpenSubtitles** v2024 film and TV subtitles; the repository has no short-row flag | 100,000 | one each | random sample, preferring rows of 3+ words | 5.7% | 10.5M |
| **NLLB (= CCMatrix)** v1 mined directly (LASER) | 900,000 | one each | top by LaBSE score, floor 0.80 | 51.6% | 117.0M |
| **MultiHPLT** v3 web text joined through English | 300,000 | one each | top by LaBSE score, floor 0.80 | 17.2% | 39.0M |
| **MultiMaCoCu** v2 web text joined through English | 1,946 | one each | all rows at or above the floor (at most 1,946) | 0.1% | 0.3M |
| **TG-10 set G** 35.7% of 4,268,571 repository rows | **1,523,490** | 1,745,034 examples, shuffled once |  | **228M** |  |
| Not in TG-10 · 2,745,081 rows go to G_REST, a pool for TG-90 |  |  |  |  |  |
| **QED** v2.0a | 0 | research-only licence |  |  |  |
| **WikiMatrix** v1 | 0 | Wikipedia-derived; kept away from FLORES+ |  |  |  |
| **wikimedia** | 0 | Wikipedia-derived; kept away from FLORES+ |  |  |  |
| **MultiCCAligned** v1.1 | 0 | built on CCAligned and joined through English; lowest-priority mined pool |  |  |  |
| **GNOME** | 0 | 110 UI strings |  |  |  |
| **NLLB (= CCMatrix)** v1 | 0 | rows ranked below the top 900,000 or under the 0.80 floor |  |  |  |
| **MultiHPLT** v3 | 0 | rows ranked below the top 300,000 or under the 0.80 floor |  |  |  |
| **OpenSubtitles** v2024 | 0 | rows outside the 100,000 sample |  |  |  |
| **MultiMaCoCu** v2 | 0 | rows under the 0.80 floor, if any |  |  |  |

- **All human-translated text, in both directions.** SETIMES, TED2020, SciPar, GlobalVoices, EUbookshop, NeuLab-TedTalks, Tatoeba and ELRC-wikipedia_health: 221,544 rows, 25.4% of TG-10's examples. SciPar, EUbookshop and NeuLab-TedTalks are new.
- **Mined text once per row, best first.** The top 900,000 NLLB pairs and the top 300,000 MultiHPLT pairs by LaBSE score, never below 0.80, plus every MultiMaCoCu pair that clears it. If fewer rows clear the floor, TG-10 takes fewer; the floor isn't lowered. NLLB (51.6%) is mined directly; MultiHPLT and MultiMaCoCu (17.3%) are joined through English, so they rank lower.
- **Some conversation.** 100,000 OpenSubtitles lines (5.7%), sampled at random and preferring lines of three words or more, one direction each.
- **Left out.** QED is research-only, and every later model inherits TG-10. WikiMatrix and wikimedia are Wikipedia text, kept away from FLORES+. MultiCCAligned is CCAligned joined through English, the weakest mined pool. GNOME is 110 interface strings.
- **One shuffle.** TG-D0 shuffles all 1.75M examples once with a fixed seed, so a checkpoint at any point has seen the same mix as the full run.

Row counts are the repository's own deduplicated counts (opus-mk-sq-clean-dedup, default config, selected by source_id), not OPUS catalogue figures. NLLB v1 repeats CCMatrix for this pair, so the two count as one source. Mined counts are targets; TG-D0 records how many rows clear 0.80.

### Is 1.75M examples the right amount? TG-10 checks

More data can make an already-tuned model worse, so TG-10 measures its own curve instead of assuming one. It trains at a constant learning rate and saves checkpoints at ≈0.6M and ≈1.0M examples. A short cooldown branched from each gives two smaller finished models for ≈3 extra GPU hours, and all three are scored side by side.

TG-10: warm-up, constant rate, cooldowncooldown branch, 10% of the steps so farsaved constant-rate checkpointscored endpoint

| Development result | TG-10 endpoint | What happens next |
| --- | --- | --- |
| The full run is best, or tied with the 1.0M branch inside the interval | **Full run**, 1.75M examples | G stays as frozen |
| The 1.0M branch beats the full run beyond the interval, in either direction | **1.0M branch** | Unused rows return to G_REST; the paper reports it as dilution |
| The full run beats 1.0M beyond the interval and the curve still climbs steeply | **Optional TG‑10‑EXT** | Resume at constant rate from the cooldown start, add next-ranked NLLB pairs above 0.80 up to ≈2.2M examples, cool down again: ≈11 h, not in the totals |

Scored on FLORES_DEV in both directions with chrF++, BLEU and COMET-22 or MetricX-24, 95% paired bootstrap; copy rate, wrong-language rate (OpenLID) and output-to-source length drift are logged as failure checks. Development data only. The chosen size and composition are frozen in the manifest before TG-20 starts.

- **Full-weight tuning suits this much data.** Zhang et al. (2024) conclude: “For million-scale datasets, FMT would be good.” [arXiv 2402.17193](https://arxiv.org/abs/2402.17193)
- **But more can hurt.** In ALMA (en→ru), LLaMA-2-7B “peaks with 10K and 100K training data before experiencing a decline” in COMET, while BLEU kept rising. [arXiv 2309.11674](https://arxiv.org/abs/2309.11674)
- **Mined data needs a filter.** Adding an unfiltered web-crawled corpus cut an NMT system from 27.2 to 17.3 BLEU (−9.9) in Khayrallah and Koehn (2018). [W18-2709](https://aclanthology.org/W18-2709/)
- **A cooldown can start late.** Constant rate plus cooldown matches cosine and lets runs of different lengths reuse one trajectory; (1−sqrt) beats a linear cooldown (Hägele et al., 2024). [arXiv 2405.18392](https://arxiv.org/abs/2405.18392)

### Legal steps: the Gazette training pool split three ways

The replay arm has to train on Gazette text no model has seen, or it would only be a second pass over TG-20's data. So the pool is split by whole issues, stratified by year, before TG-20 starts.

| Slice | Rows | Of 290,000 | Of training pool | Used by | Purpose |
| --- | --- | --- | --- | --- | --- |
| GAZ_DEV | 13,200 | 4.6% | — | selection in every run | Issue-disjoint validation split |
| GAZ_TEST | 10,000 | 3.4% | — | scoring every model | Issue-disjoint test split |
| Removed | 16,800 | 5.8% | — | nothing | Training pairs that repeat a dev or test sentence |
| L_MAIN | 170,000 | 58.6% | 68% | TG-20, TG-21 | Legal learning, trained once |
| L_ARMS | 40,000 | 13.8% | 16% | every TG-30 arm; TG-90 | Shared legal stream of the arms |
| L_FRESH | 40,000 | 13.8% | 16% | TG-31 (matched amount); TG-90 | New parallel sentences for the replay arm |

Rows are planned shares of slvesnik-mk-sq-clean, assuming its issue-disjoint dev and test splits keep their earlier sizes; TG-D0 records the real counts. By TG-90 every training row has been used: L_MAIN in TG-20, the rest in TG-90.

### Word lists

| Source | Format | Entries | Licence | Goes into |
| --- | --- | --- | --- | --- |
| **VERBIS termbase** private repo | Term + definition, both languages; four domains | 15,925 | Redistribution restricted | V\_\* tasks, all arms |
| **Media Literacy Dictionary** local | Term + definition, both languages | 1,081 valid | No licence found; treated as copyrighted | V\_\* tasks, fifth domain |
| **MUSE dictionary** EdonFetaji/mk-sq-dictionary-muse | Word pairs built through English | 7,388 | CC BY-NC 4.0 | W_GEN, TG-36 |
| **Goethe Verlag phrasebook** local | Conversational phrase pairs | 1,779 unique | All rights reserved; scraping prohibited | W_GEN, TG-36, research only |
| **Articles and notices** local, 500k rows | Legal text mapped to articles and notices | 500,000 | Own scrape | A_NOTICES, TG-90 only if it's paired |
| **Shqipopedia; Goethe thematic vocabulary** | Glossary rows; word lists | 696 · 1,904 | — | Not used |

Before any word task, 1,701 of the 17,006 definition-lexicon entries (10% of concepts, all senses and both directions together) go into V_HOLDOUT, which no model trains on before TG-90.

### What each run trains on

General · human translationGeneral · NLLBGeneral · via EnglishGeneral · subtitlesGeneral replayGazetteFresh GazetteWord taskGeneral word pairs

| Run | Token mix | Tokens | Training |
| --- | --- | --- | --- |
| TG-10 General MK–SQ |  | 228M | 25 h |
| TG-20 Shared legal |  | 170M | 19 h |
| TG-21 Gazette only |  | 51M | 5.7 h |
| TG-31 Fresh parallel sentences |  | 32M | 3.6 h |
| TG-32 Term pairs |  | 28M | 3.1 h |
| TG-33 Term pairs + definitions |  | 32M | 3.6 h |
| TG-34 Definition translation |  | 32M | 3.6 h |
| TG-35 Shuffled definitions |  | 32M | 3.6 h |
| TG-36 General word pairs |  | 31M | 3.5 h |
| TG-90 Final, minimum |  | 73M | 8.1 h |

Every arm's shared stream is L_ARMS plus an equal amount of general replay, 24M tokens. TG-90 is shown at its minimum; adding G_REST takes ≈40 h more.

### Dataset tags

| Tag | Built from | Size | Used by |
| --- | --- | --- | --- |
| General |  |  |  |
| G | Every human-translated row, both directions; NLLB top 900k and MultiHPLT top 300k by LaBSE (≥ 0.80); MultiMaCoCu above the floor; OpenSubtitles 100k. Shuffled once | 1,523,490 rows | TG-10 |
| G_REPLAY | Examples resampled from G in its source proportions, fixed per seed; TG-20's replay is ≈52% of G | 119M in TG-20 | TG-20, every arm, TG-90 |
| G_REST | Every repository row not in G: QED, WikiMatrix, wikimedia, MultiCCAligned, GNOME, lower-ranked NLLB and MultiHPLT, the rest of OpenSubtitles | 2,745,081 rows | TG-90, optional |
| Legal |  |  |  |
| L_MAIN | 68% of the Gazette training issues | 170,000 rows | TG-20, TG-21 |
| L_ARMS | 16% of the issues, unseen by TG-20 | 40,000 rows | Every arm; TG-90 |
| L_FRESH | 16% of the issues, unseen by any model until TG-31 | 40,000 rows | TG-31 (28,161 rows); TG-90 |
| A_NOTICES | Articles-and-notices corpus, dev and test issues removed | 500,000 rows | TG-90, if paired |
| Word tasks · VERBIS + Media Literacy, 15,305 training entries, 3 presentations per direction |  |  |  |
| V_TERM | Source term → target term | 91,830 examples | TG-32, TG-36 |
| V_DEF_CTX | Source term + source-language definition → target term | 91,830 examples | TG-33 |
| V_DEF_PAR | Definition → definition in the other language, LaBSE ≥ 0.70 | 14,716 pairs | TG-34 |
| V_DEF_SHUF | V_DEF_CTX with definitions swapped between similar-length entries of one domain | 91,830 examples | TG-35 |
| V_HOLDOUT | 10% of concepts, never trained before TG-90 | 1,701 entries | Scored as TERM_HOLDOUT |
| V_ALL | Every entry, hold-out included, in the winning task format | 17,006 entries | TG-90 |
| W_GEN | MUSE word pairs and Goethe phrases, no definitions | 7,388 + 1,779 | TG-36; TG-90 if TG-36 helped |

07 · ORDER OF WORK

## What runs first, and what waits for results

Runs go in priority order on the two GPUs. Priority 1 answers the central question; priority 2 explains it; priority 3 extends it; TG-90 waits for all of them. The core study needs about 85 hours, 3.5 days of continuous GPU time, including the ≈3 h size check.

day 1day 2day 3day 4day 5

- Data freeze and baseline2 h
- TG-10 general26 h
- TG-10 size check3 h
- TG-20 legal19 h
- Arms, seed 112 h
- Arms, seeds 2 and 324 h
- Priority 2: TG-34, TG-21, TG-3514 h
- Priority 3: TG-36 and extra seeds20 h
- TG-90 final (minimum)9 h

Hours include scoring each model (≈0.5 h with vLLM and greedy decoding). Restarts, checkpoint saves and debugging usually add 30–50% of calendar time.

| # | Run | Priority | Starts after, or condition | Hours | Running total |
| --- | --- | --- | --- | --- | --- |
| 1 | TG-D0 | Setup | first | 1.0 | 1 h · day 0.0 |
| 2 | TG-00 | Setup | after TG-D0 | 0.5 | 2 h · day 0.1 |
| 3 | TG-10 | 1 · core | after TG-D0 | 25.8 | 27 h · day 1.1 |
| 4 | TG-10-CD06 | 1 · core | after TG-10 | 1.2 | 28 h · day 1.2 |
| 5 | TG-10-CD10 | 1 · core | after TG-10 | 1.8 | 30 h · day 1.3 |
| 6 | TG-10-EXT | 1 · core | only if the full run beats the 1.0M branch beyond the interval and the curve still climbs steeply; not counted in totals | 10.7 | not counted |
| 7 | TG-20 | 1 · core | after TG-10, TG-10-CD06, TG-10-CD10 | 19.4 | 50 h · day 2.1 |
| 8 | TG-31-s1 | 1 · core | after TG-20 | 4.1 | 54 h · day 2.2 |
| 9 | TG-32-s1 | 1 · core | after TG-20 | 3.6 | 57 h · day 2.4 |
| 10 | TG-33-s1 | 1 · core | after TG-20 | 4.1 | 62 h · day 2.6 |
| 11 | TG-31-s2 | 1 · core | after TG-20 | 4.1 | 66 h · day 2.7 |
| 12 | TG-32-s2 | 1 · core | after TG-20 | 3.6 | 69 h · day 2.9 |
| 13 | TG-33-s2 | 1 · core | after TG-20 | 4.1 | 73 h · day 3.1 |
| 14 | TG-31-s3 | 1 · core | after TG-20 | 4.1 | 77 h · day 3.2 |
| 15 | TG-32-s3 | 1 · core | after TG-20 | 3.6 | 81 h · day 3.4 |
| 16 | TG-33-s3 | 1 · core | after TG-20 | 4.1 | 85 h · day 3.5 |
| 17 | TG-34-s1 | 2 · interpret | after TG-20 | 4.1 | 89 h · day 3.7 |
| 18 | TG-21 | 2 · interpret | after TG-10 | 6.2 | 95 h · day 4.0 |
| 19 | TG-35-s1 | 2 · interpret | only if TG-33 beats TG-32 on reference-confirmed legal term success, mean of 3 seeds | 4.1 | 100 h · day 4.1 |
| 20 | TG-36-s1 | 3 · extend | after TG-20 | 4.0 | 104 h · day 4.3 |
| 21 | TG-34-s2 | 3 · extend | only if seed 1 differs from TG-31 by more than the seed-to-seed spread of the core arms | 4.1 | 108 h · day 4.5 |
| 22 | TG-34-s3 | 3 · extend | only if seed 1 differs from TG-31 by more than the seed-to-seed spread of the core arms | 4.1 | 112 h · day 4.7 |
| 23 | TG-36-s2 | 3 · extend | only if seed 1 differs from TG-32 by more than the seed-to-seed spread of the core arms | 4.0 | 116 h · day 4.8 |
| 24 | TG-36-s3 | 3 · extend | only if seed 1 differs from TG-32 by more than the seed-to-seed spread of the core arms | 4.0 | 120 h · day 5.0 |
| 25 | TG-90 | Final | after every core run is scored; recipe follows the results | 8.6 | 128 h · day 5.3 |

08 · EXPERIMENTS

## Every run, step by step

Nothing in a later step may change a decision frozen in an earlier one.

Step 0 · Setup

### Freeze the data, score the baseline

TG-D0 · DATA

### Freeze the data

Setup

CPU + ≈1 GPU hour

Produces

Every dataset tag in section 06, each with a manifest and file hash.

Steps

Select G by source_id; LaBSE-score every NLLB, MultiHPLT and MultiMaCoCu pair (none has an embedding filter yet) and store the score and an english_mediated flag; record how many pass 0.80; check the release scope of SciPar and EUbookshop; shuffle TG-10's examples once with a fixed seed; LaBSE-score definition pairs; split the Gazette training pool by issue into L_MAIN, L_ARMS and L_FRESH; remove training pairs that repeat a dev or test sentence; hold out 10% of lexicon concepts; mask definitions that contain the target term; build the terminology test subsets.

Why

Every later run reads the same frozen files, and no test sentence can leak into training.

TG-00 · BASE

### Original TranslateGemma

Scored only

≈0.5 h

Why

The reference for every gain and loss. TranslateGemma's report lists English↔Macedonian and English↔Albanian but not the direct pair, so direct MK–SQ quality is measured, not assumed.

Step 1 · General

### Teach direct Macedonian ↔ Albanian

TG-10 · GEN

### General MK–SQ

Saved endpointPriority 1

1 run · ≈25 h + ≈3 h size check

Starts from

TG-00

Trains on

G: every human-translated row in both directions (221,544), plus the top 900,000 NLLB and 300,000 MultiHPLT pairs by LaBSE (≥ 0.80), MultiMaCoCu pairs above the floor and 100,000 OpenSubtitles lines, one direction each

Size

1,523,490 rows (35.7% of the deduplicated set), 1.75M examples, 228M tokens

Schedule

Constant rate after warm-up, (1−sqrt) cooldown over the final ≈15%; constant-rate checkpoints at ≈0.6M, ≈1.0M and the cooldown start (≈1.48M)

Decides

Its own endpoint, by the [size check](#size), frozen before TG-20

Why

Teaches the direct pair once. Its general scores are the reference for RQ2.

TG-10-CD06 · CD10

### Size-check branches

Priority 1EXT conditional

2 short runs · ≈3 h

Starts from

TG-10's constant-rate checkpoints at ≈0.6M and ≈1.0M examples

Trains on

The next 10% of TG-10's own shuffled stream, cooling the rate to zero

Compares

Endpoints at ≈0.66M, ≈1.1M and 1.75M examples on FLORES_DEV, both directions

Optional

TG-10-EXT continues past 1.75M with next-ranked NLLB pairs, only under the third rule in the [size check](#size) (≈11 h, outside the totals)

Step 2 · Legal

### Teach legal language once

TG-20 · LEG

### Shared legal

Saved endpointPriority 1

1 run · ≈19 h

Starts from

TG-10, at the endpoint the size check picks

Trains on

L_MAIN, 170,000 Gazette rows in both directions (51M tokens, 30%), and G_REPLAY from G (119M, 70%, about 52% of G)

Why

Legal learning happens once, and every arm branches from here. L_ARMS and L_FRESH stay unseen.

Answers

RQ2 TG-20 vs TG-10 shows the general cost of legal training

TG-21 · GAZ

### Gazette-only diagnostic

Priority 2

1 run · ≈6 h

Starts from

TG-10

Trains on

L_MAIN alone: TG-20's legal data without replay

Why

Repeats the NLLB E02 setting at 12B. TG-21 vs TG-20 shows how much replay protects.

Answers

RQ2

Step 3 · TG-30 arms

### Compare ways of spending the same GPU time

Every arm starts from TG-20 and trains on the same shared stream: L_ARMS in both directions plus as much general replay, 24M tokens in all, in the same order for a given seed. Each arm then adds one thing. Seed 1 of every core arm runs first.

TG-31 · REPLAY

### Fresh parallel sentences

Priority 1

3 seeds · ≈3.6 h each

Adds

28,161 new rows from L_FRESH in both directions, 8.4M tokens: exactly the compute of TG-33's word task

Why

Plain parallel-corpus training, the realistic alternative to building a dictionary. The word-task arms have to beat this run, not TG-20.

Answers

RQ1Central (b)RQ2

TG-32 · TERMS

### Term pairs

Priority 1

3 seeds · ≈3.1 h each

Adds

V_TERM: 15,305 entries from the four VERBIS domains and the Media Literacy Dictionary, each shown 3 times in each direction (91,830 examples, 4.3M tokens)

Why

Tests whether knowing the term pairs helps, and is the baseline the definitions arm must beat.

Answers

RQ1Central (a)Central (b)RQ3RQ4

TG-33 · TERMS-DEF

### Term pairs with definitions

Priority 1

3 seeds · ≈3.6 h each

Adds

V_DEF_CTX: TG-32's entries and presentations, with the source-language definition added to each input. The output is still just the target term. 8.4M tokens.

Why

The central test. Everything except the definition matches TG-32.

Answers

Central (a)Central (b)RQ1RQ3RQ4

TG-34 · DEF-TRANS

### Definition translation

Priority 2

seed 1; seeds 2–3 if warranted

Adds

V_DEF_PAR: 14,716 definition pairs with LaBSE ≥ 0.70, translated both ways (1.7 passes), matched to TG-33's 8.4M tokens

Why

Isolates the definition-translation half of the NLLB E01 recipe, the part that may have protected general quality.

Answers

RQ2Central (b)

TG-35 · SHUF-DEF

### Shuffled definitions

Priority 2

1 seed, conditional

Runs if

TG-33 beats TG-32 on reference-confirmed legal term success, averaged over three seeds

Adds

V_DEF_SHUF: TG-33's entries, each given the definition of another entry of similar length

Why

Shows whether the definition's meaning mattered or only the extra input text.

TG-36 · GEN-LEX

### General word pairs

Priority 3

seed 1; seeds 2–3 if warranted

Adds

TG-32's V_TERM plus W_GEN: 7,388 MUSE pairs and 1,779 Goethe phrases, 3 times each direction (3.0M tokens)

Why

Tests whether everyday word knowledge without definitions helps general translation. Its research-only data stays out of the shared checkpoints and the core results.

Answers

RQ5Central (b)

Step 4 · Final

### Build the strongest model the evidence supports

TG-90 · FINAL

### All-data model

After results

1 run · ≈9 h minimum

Starts from

TG-20

Trains on

L_ARMS and L_FRESH in both directions, so the whole Gazette training pool is used; V_ALL in the winning task format; W_GEN only if TG-36 beat TG-32; general replay equal to the rest

Options

G_REST, the 2,745,081 repository rows not in TG-10 (≈40 h), and A_NOTICES if it's paired (≈27 h per direction)

Release

A public checkpoint inherits everything TG-10 and TG-20 trained on, so TED2020, NeuLab-TedTalks, SciPar, EUbookshop, NLLB and OpenSubtitles need their licences checked, VERBIS needs a permission basis, and Goethe and the Media Literacy Dictionary stay out. Its scores show the model is good; they don't answer the research questions.

09 · FAIRNESS

## What stays equal across the arms

- **Same start.** Every arm loads TG-20 and uses the same optimizer, learning rate, schedule and batch size.
- **Same shared stream.** L_ARMS and the same general replay, in the same order for a given seed.
- **Same entries.** TG-32, TG-33 and TG-35 show the same 15,305 entries 3 times in each direction. Only the input differs.
- **Same GPU time where it matters.** TG-31, TG-34 and TG-35 process exactly as many tokens as TG-33, so plain parallel training gets the same compute as the definitions arm. TG-32 processes fewer, because bare terms are short; that only favours its rivals.
- **Really new legal text.** L_FRESH comes from Gazette issues no model has seen, so TG-31 tests more parallel data, not a second pass.
- **A real hold-out.** 1,701 lexicon entries (V_HOLDOUT) never appear in any word task, which shows whether word training helps or hurts terms it never saw.
- **No peeking.** Settings and checkpoints are chosen on development data only. The extra input tokens in TG-33 are reported as compute cost.

10 · SCORING

## How every model is scored

No human reviewers are available, so every measure is automatic and reproducible. Each model's translations are generated once, and every metric reads those same outputs.

| Tag | Built from | Size | Used for |
| --- | --- | --- | --- |
| Development · selection only |  |  |  |
| GAZ_DEV | Gazette validation split, issue-disjoint | ≈13,200 | Checkpoint selection |
| FLORES_DEV | FLORES+ dev, mkd_Cyrl–als_Latn | 997 | Selection and the retention rule |
| TERM_DEV | GAZ_DEV sentences with V_TERM terms, reference-confirmed | counted at TG-D0 | Spot checks during the arms |
| Test · reported results |  |  |  |
| GAZ_TEST | Gazette test split, issue-disjoint | ≈10,000 | Legal quality |
| GAZ_TEST_NOVEL | GAZ_TEST minus templates and near-duplicates of training sentences | counted at TG-D0 | Legal quality without template inflation |
| FLORES_DEVTEST | FLORES+ devtest, mkd_Cyrl–als_Latn | 1,012 | General quality, primary |
| NTREX | NTREX-128 Macedonian and Albanian references | 1,997 | General quality, continuity with the pilot |
| TERM_LEG | GAZ_TEST sentences with legal V_TERM terms | counted at TG-D0 | Central (a), RQ1, RQ3 |
| TERM_NONLEG | Test sentences with non-legal and media-literacy terms | counted at TG-D0 | RQ4 |
| TERM_GEN | FLORES+ and NTREX sentences with W_GEN words | counted at TG-D0 | RQ5 |
| TERM_HOLDOUT | Test sentences with V_HOLDOUT terms | counted at TG-D0 | Untrained terms; not reported for TG-90 |

- **Term success.** A term counts as correct when the output contains the listed target, matched on its first max(4, length − 2) characters, which accepts inflected forms. The headline figure uses only occurrences where the human reference itself uses that target.
- **Translation quality.** chrF++ and BLEU through sacreBLEU with exact signatures, plus number fidelity and a negation check.
- **Uncertainty.** Paired bootstrap with 10,000 resamples by document, 95% intervals, and seed variation reported separately.

11 · OPEN ITEMS

## To settle before TG-D0 finishes

- **Articles and notices (500k rows):** paired Macedonian–Albanian text, or one language? It stays out of TG-10 to TG-30 either way. If it comes from the same Gazette issues, dev and test issues are removed before any use.
- **SciPar and EUbookshop release scope:** both are new in TG-10, and TG-D0 checks their release_scope. If either turns out research-only, the rule that keeps QED out of TG-10 would apply to it too.
- **Gazette splits:** confirm that slvesnik-mk-sq-clean keeps the issue-disjoint dev and test splits (about 13,000 and 10,000 rows) that the numbers here assume.
- **Fourth VERBIS domain:** the earlier count names it health (2,059 entries); the latest inventory says judicial. RQ4 reports by domain, so the label matters.
- **GPU link:** with NVLink, FSDP's weight exchanges cost little; over PCIe, keep micro-batches near 8k tokens per GPU.

Machine-readable twins: `tg_experiment_plan.csv` (25 runs, in run order, with priority, data spec, tokens and hours) and `tg_datasets.csv` (54 rows, one per source inside each dataset tag). Both use the tags on this page.

NLLB pilot figures from the E00–E02 experiment report. TranslateGemma results are prospective; token counts and hours are estimates until count_tokens.py and TG-10's training log measure them.