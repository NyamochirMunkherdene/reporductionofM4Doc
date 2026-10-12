# M4Doc Reproduction Study

An independent experimental reproduction study of **M4Doc**, covering a text-only English-to-Chinese translation model, a Nougat-based multimodal baseline, and an M4Doc experiment using Vary-Toy.

> **Status:** Experiments concluded. This is a partial reproduction study, **not** a verified reproduction of all results reported in the paper. Reported-paper values below are transcribed from the author's research notes and have not been independently checked against the publication.

## Overview

The project explored whether the available methodological information was sufficient to reproduce reported results. The study included:

1. Pretraining a text-only English-to-Chinese translation model on the UN Parallel Corpus, followed by fine-tuning on DoTA.
2. Constructing and training a baseline with a Nougat encoder, projection layer, and text-only decoder.
3. Experimenting with an M4Doc architecture using a frozen Vary-Toy MLLM encoder, semantic and Nougat encoders, projection/alignment layers, and a pretrained EN-to-ZH translation decoder.

The original implementation and some experimental details were not available to the researcher, so parts of the pipeline and training configuration were reconstructed. Differences from the reported results should not be attributed to any single cause without further controlled experiments.

## Dataset and evaluation

- **UN Parallel Corpus (English–Chinese):** text-only pretraining.
- **DoTA:** fine-tuning and multimodal experiments. The dataset was received via Hugging Face, with access confirmed by email.
- **Metrics recorded:** BLEU, BLEU-PT, STEDS, training loss, and evaluation loss.

DoTA fine-tuning notes list **124,338 training samples** and **1,004 validation samples**. Dataset access and redistribution rights should be checked before publishing any data.

## Results at a glance

| Experiment | Checkpoint | BLEU | BLEU-PT | STEDS |
|---|---:|---:|---:|---:|
| Text-only, reported in paper | — | 47.61 | 54.16 | 92.89 |
| Text-only, reproduced | 9,000 | 31.64 | 33.18 | *see note* |
| Text-only, reproduced | 38,000 | 44.37 | 48.45 | *see note* |
| Text-only, reproduced | 48,000 | 44.90 | 49.60 | *see note* |
| Text-only, reproduced | 78,000 | 45.05 | 47.16 | 80.75 |
| Nougat-based baseline, reported in paper | — | 37.60 | 61.52 | 83.08 |
| Nougat-based baseline, reproduced | 47,750 | 42.64 | 42.78 | 87.75 |

**STEDS evaluation caveat:** Early text-only evaluations at checkpoints 9,000, 38,000, and 48,000 all showed STEDS **29.24**. The researcher traced this repeated value to a decoder that flattened or removed `\n\n` formatting. Those early STEDS values are therefore omitted from the comparison as unreliable; the later checkpoint-78,000 value is recorded separately. The notes do not establish whether all other evaluation differences were resolved.

## 1. Text-only model

### 1.1 Pretraining on UN Parallel Corpus (En–Zh)

The notes reference the text-only MT approach of *Attention Is All You Need* (Vaswani et al., 2017). The specified settings available to the researcher were **100,000 maximum steps**, **7e-5 learning rate**, **4,096 maximum tokens per batch**, and **0.05 warmup ratio**. Other unspecified choices were made by reference to that paper; they should not be interpreted as confirmed M4Doc settings.

| Setting | Recorded value |
|---|---|
| Dataset | UN Parallel Corpus (English–Chinese) |
| Optimizer | AdamW |
| Scheduler | Linear |
| Learning rate | 7e-5 |
| Maximum training steps | 100,000 |
| Warmup ratio | 0.05 |
| Global / per-GPU batch size | 16 / 4 |
| GPUs / gradient accumulation | 4 / 1 |
| Precision | BF16 |
| Model max length / data max length | 2,048 / 512 |
| Initial / final training loss | ~10 / ~7 |
| Recorded training time | ~1 h 36 min |

An evaluation recorded **BLEU 0.13**, **BLEU-PT 0.02**, and **STEDS 29.24**. An example output repeated the Chinese character `的` rather than generating a useful translation. A separate learning-rate experiment at **1e-4** took approximately **1 h 36 min**; the final loss and comparable evaluation metrics were not recorded in the supplied notes.

Evidence: [pretraining evaluation](https://gyazo.com/e762b4d6300007c96c0b31aa1334a78f), [additional run screenshot](https://gyazo.com/e7150191ec137cc8f568edc0d9217007), [1e-4 run screenshot 1](https://gyazo.com/44f18ba123466d687ea4cd963738413f), [1e-4 run screenshot 2](https://gyazo.com/89972ac9b1f76214a6e98495dc3ff18d).

### 1.2 Fine-tuning on DoTA

| Setting | Recorded value |
|---|---|
| Starting checkpoint | UN-pretrained text-only model |
| Dataset | DoTA |
| Training / validation samples | 124,338 / 1,004 |
| Epochs | ~38 |
| Steps | ~74,000 |
| Learning rate | 5e-5 |
| Maximum sequence length | 1,536 |
| Batch size | 64 |
| Precision | BF16 |
| Initial / final train loss | ~8 / 0.4328 |
| Final validation loss | 0.4834 |

An earlier log at **25 epochs / 48,550 steps** recorded train loss **0.5779**, evaluation loss **0.5747**, and approximately **7.4 hours** of training. The notes suggest training could potentially have stopped around 50,000 steps; this was an observation, not a validated stopping-point study.

The text-only benchmark comparison appears in the results table above.

Evidence: [fine-tuning screenshot 1](https://gyazo.com/f2879a34465f15733c59f22757c9de4c), [fine-tuning screenshot 2](https://gyazo.com/fc5edac26f743d20bce61e43fd4c4a33).

## 2. Nougat-based baseline

Architecture: **Nougat encoder → projection layer → text-only decoder**.

| Setting | Recorded value |
|---|---|
| Dataset | DoTA |
| GPUs | 4 |
| Maximum sequence length | 1,536 |
| Maximum training steps | 100,000 |
| Initial learning rate | 5e-5 |
| Scheduler / warmup | Linear / 1,000 steps |
| Weight decay | 0.01 |
| Per-GPU batch size | 4 |
| Gradient accumulation | 8 |
| Effective global batch | 128 |
| Optimizer | AdamW |
| Precision / TF32 | BF16 / enabled |
| Gradient checkpointing | Enabled |
| Logging interval | 10 steps |
| Initial / final training loss | 6.393 / 0.6912 |
| Evaluation loss | 0.7139 |
| Logged learning rate at epoch 12.33 | 2.70445e-5 |
| Recorded training time | ~24 hours |

At **checkpoint 47,750**, the recorded evaluation was **BLEU 42.64**, **BLEU-PT 42.78**, and **STEDS 87.75**. Compared with the paper values transcribed in the notes, BLEU and STEDS were higher while BLEU-PT was lower. The researcher stopped this run without determining whether additional training would improve the overall comparison.

Evidence: [baseline screenshot 1](https://gyazo.com/53df968552c2062b942a4ced72c0d379), [baseline screenshot 2](https://gyazo.com/56c12e857dc9798ce6f120f165e7dc52).

## 3. M4Doc experiment with Vary-Toy

**Important architectural limitation:** The notes indicate that the intended full M4Doc reproduction uses **Vary base**, while this experiment used **Vary-Toy**. Obtaining Vary base would have required reproducing the separate [Vary paper](https://github.com/Ucas-HaoranWei/Vary/blob/main/Vary_paper.pdf). Accordingly, the run below should **not** be presented as a faithful full M4Doc reproduction.

Recorded architecture:

- Frozen Vary MLLM encoder (teacher)
- Semantic encoder
- Nougat encoder
- Projection / alignment layers
- Pretrained English-to-Chinese translation decoder

| Setting | Recorded value |
|---|---|
| Dataset | DoTA |
| GPUs / machines | 4 / 1 |
| Global / per-GPU batch size | 64 / 2 |
| Gradient accumulation | 8 |
| Maximum / completed steps | 30,000 / 30,000 |
| Learning rate | 5e-5 |
| Scheduler / warmup | Linear / 1,000 steps |
| Optimizer | AdamW |
| Precision | BF16 |
| Evaluation / checkpoint interval | 1,000 steps each |
| Dataloader workers | 16 |
| MLLM encoder | Frozen |
| KD loss weight | 1.0 |
| Initial / final training loss | 6.3496 / 1.2789 |
| Evaluation loss | 1.1708 |
| Runtime | 19:36:02 |
| Final logged KD loss | 0.1712 |
| Final logged learning rate | 0.0 |
| Final logged epoch | 15.44 |
| Logged overall train loss | 1.7933 |

**No BLEU, BLEU-PT, or STEDS values for this Vary-Toy run were provided in the notes.** The logged loss values alone cannot establish equivalence to the published M4Doc results.

Evidence: [Vary-Toy training screenshot 1](https://gyazo.com/aa83accbd13a1489e44762721ed13551), [Vary-Toy training screenshot 2](https://gyazo.com/d5284e11a88ef529168f5dc4e0329ad2).

## Reproducibility limitations and lessons

- The researcher's notes state that original implementation code and some methodological details were unavailable, requiring an independently constructed pipeline and assumptions for unspecified settings.
- Pretraining at the initially recorded learning rate produced poor sample translations; a separate learning-rate trial was performed, but its outcome is not fully documented in the notes.
- An evaluation decoder flattened `\n\n` formatting, producing repeated, misleading STEDS scores until the issue was identified.
- Text-only and Nougat-based baseline results did not consistently match the reported benchmark metrics.
- The Vary-Toy experiment is not equivalent to the Vary-base configuration identified as necessary for the full M4Doc reproduction.

These observations document an experimental attempt and its limitations; they do not independently demonstrate that the original paper's methods or reported results are incorrect.

## Repository contents

> Add links to actual source files, scripts, configuration files, and instructions once they are uploaded. Do not claim code is available until it is present in the repository.

Suggested additions:

- Training and evaluation scripts
- Model configuration files and dependency versions
- Instructions for preparing authorized datasets
- Checkpoint-specific metric tables and sample outputs
- Hardware and software environment details
- Notes on newline handling in the STEDS evaluation pipeline

Avoid committing private credentials, restricted datasets, or large checkpoints without confirming redistribution rights.

## References and acknowledgments

- M4Doc: add the full paper citation and official project URL.
- Vaswani et al. (2017), *Attention Is All You Need*.
- [Vary paper (repository PDF)](https://github.com/Ucas-HaoranWei/Vary/blob/main/Vary_paper.pdf).
- DoTA dataset: add the official citation and permitted access link.

This repository is an **independent study**, not an official M4Doc implementation.
