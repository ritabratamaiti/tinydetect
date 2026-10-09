# TinyDetect

AI text and image detection that runs in your browser. The models are downloaded once and run locally with ONNX Runtime Web (WebGPU when available, WebAssembly otherwise), so nothing you check is sent anywhere.

**Demo:** https://ritabratamaiti.github.io/tinydetect/

| | Starting model | Full size | Shipped size | Time per check |
|---|---|---|---|---|
| Text | [Fakespot RoBERTa-base detector](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1) (Apache-2.0) | 499 MB | 97 MB (89 MB compressed) | ~250 ms WebGPU, 2–6 s CPU |
| Image | [Community Forensics ViT-S/16, 384 px](https://huggingface.co/OwensLab/commfor-model-384) (MIT) | 87 MB | 26 MB (25 MB compressed) | ~120 ms WebGPU, 1–3 s CPU |

Times are from a laptop with an RTX 4050 while other jobs were running.

## How it works

Both detectors are existing, well-tested models. Most of the work here is making them small and fast enough for a web page without changing their answers.

**Choosing the models.** I scored ten public detectors (four text, six image) on two sets of fresh samples (below). Fakespot was the best text detector by a wide margin. Community Forensics, trained on images from about 4,800 generators, was close to the best image detectors at a quarter of their size and never flagged a real photo.

**Light tuning (text).** Fakespot flags a lot of formal human writing: 28% of held-out human text at the 0.6 cut-off, mostly Wikipedia- and novel-style prose. I fine-tuned it briefly on matched human and AI text (HC3, GPT-wiki-intro, Project Gutenberg, HAP-E, WildChat, COLING-2025 MGT and casual messages from Gemma 3 1B), weighting human errors twice and keeping AI-sample outputs close to the original. Full tuning fixed the false flags but lost some detection of casual AI writing, so the shipped weights are a 50/50 average of the original and tuned models (WiSE-FT). On the held-out split that cut formal human flags from 36% to 3% and casual human flags from 23% to 7%, with AI detection unchanged (91% casual, 97% formal).

**Calibration (image).** Community Forensics ranks images well but is cautious: many AI images scored under 0.5. A two-parameter rescaling of its logit (`p = sigmoid(0.587·logit + 2.969)`), fitted on 500 separate calibration images with real photos weighted three times, fixes that. The text model is left uncalibrated, because rescaling cut its detection of casual AI text.

**Text compression.**
- Word embeddings (50,265 × 768) are taken out of the graph and stored as int4 with one fp16 scale per 32 values (21.7 MB). JavaScript decodes the rows it needs and feeds them to ONNX as `inputs_embeds`.
- I measured how much each of the 72 weight matrices moves the output when quantised alone. The attention value and output projections are the most sensitive, and the feed-forward layers tolerate 4 bits well. The 24 most sensitive matrices use 8-bit blocks; the other 48 use 4-bit blocks (`MatMulNBits`, block 32). That brings the transformer from 344 MB to 75 MB.
- A byte-level BPE tokenizer and Fakespot's own `clean_text` preprocessing are reimplemented in JavaScript. Both match the Python versions exactly on every test input.

**Image compression.** All transformer weights use 8-bit blocks (87 MB → 26 MB) with no loss. 4-bit was smaller (18 MB) but less accurate and 45 times slower on WebGPU. Shrinking the input below 384 px cost too much accuracy (calibration AUROC fell from 0.992 to 0.919 at 256 px).

**Runtime.** `boot.js` loads the WebGPU build of ONNX Runtime only if the browser has a GPU adapter, and the smaller WASM build otherwise. A service worker adds cross-origin isolation headers, so WASM can use four threads on GitHub Pages, and caches the model files so later visits load instantly. One warm-up run compiles the GPU shaders before your first check.

| Compression | Text size | Text AUROC, set 1 / 2 | Image size | Image AUROC, set 1 / 2 |
|---|---|---|---|---|
| Full precision | 366 MB | 0.982 / 0.917 | 87 MB | 0.987 / 0.957 |
| 8-bit | 122 MB | 0.982 / 0.924 | **26 MB** | **0.989 / 0.962** |
| Mixed 4/8-bit | **97 MB** | **0.984 / 0.927** | – | – |
| 4-bit | 77 MB | 0.974 / 0.831 (original weights) | 18 MB | 0.991 / 0.948 |

## Benchmarks

Scores below are from the shipped models running in a browser. Bands: AI-generated ≥ 0.9, Possibly AI 0.6–0.9, Unclear 0.3–0.6, Human < 0.3.

**Fresh samples, set 2.** Human text: Hacker News comments from 2014, Project Gutenberg, Wikipedia as of 2019. AI text: 16 passages written by Claude, most of them casual (short emails, a Slack message, Reddit advice). Real photos: Wikimedia Commons uploads before 2019. AI images: Midjourney, DALL·E 3 and Flux.

| | n | AI-generated | Possibly AI | Unclear | Human |
|---|---|---|---|---|---|
| Human writing | 24 | 0 | 1 | 2 | 21 |
| AI writing | 16 | 5 | 6 | 2 | 3 |
| Real photos | 21 | 0 | 1 | 2 | 18 |
| AI images | 31 | 20 | 6 | 2 | 3 |

**Fresh samples, set 1.** Hacker News comments from 2012, Project Gutenberg, Wikipedia as of 2019; 16 Claude passages in mixed styles; Wikimedia Commons photos; mostly Midjourney images.

| | n | AI-generated | Possibly AI | Unclear | Human |
|---|---|---|---|---|---|
| Human writing | 24 | 0 | 2 | 2 | 20 |
| AI writing | 16 | 14 | 1 | 0 | 1 |
| Real photos | 30 | 0 | 1 | 1 | 28 |
| AI images | 50 | 43 | 6 | 0 | 1 |

**Compared with the previous version** (small models distilled from other detectors), across both sets:

| | Previous | Now |
|---|---|---|
| Human writing flagged (Possibly AI or above) | 5 of 48 | 3 of 48 |
| AI writing caught | 17 of 32 | 26 of 32 |
| Real photos flagged | 1 of 51 | 2 of 51 |
| AI images caught | 61 of 81 | 75 of 81 |

**What it misses.** Very short, casual AI text (a 90-word email, Reddit advice, a Slack message) still comes back as human or unclear. A few 2019 Wikipedia paragraphs score as possibly AI, and some Hacker News comments land in Unclear. On images, most misses are DALL·E 3 pictures from one collection.

**Caveats.** Neither model was trained on the fresh samples, but set 2 was looked at when choosing the 50/50 blend, so it isn't a perfectly blind test. Sets are small (40 texts and 52–80 images each), so a single sample moves a percentage a lot. Paraphrasing tools, heavy editing, screenshots and new generators can fool both models. Per-sample scores are in `training/`.

## Reproduce

`training/` has the scripts: candidate benchmarks (`bench_candidates.py`), text tuning (`tune_fakespot.py`, `wise_fakespot.py`, `pick_alpha.py`, `make_wise.py`), export and quantisation (`export_text.py`, `mixed_text.py`, `mix_apply.py`, `optimise_image.py`, `quant_sweep_image.py`) and evaluation (`eval_text_variants.py`). Everything runs on one laptop.

## Credits and licences

- Text model: [fakespot-ai/roberta-base-ai-text-detection-v1](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1) (Apache-2.0), based on [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base). The preprocessing in `app.js` is ported from its `utils.py`.
- Image model: [OwensLab/commfor-model-384](https://huggingface.co/OwensLab/commfor-model-384), [Community Forensics](https://github.com/JeongsooP/Community-Forensics) (MIT, © 2025 Jeongsoo Park).
- Tuning and calibration data: Hello-SimpleAI/HC3, aadityaubhat/GPT-wiki-intro, Project Gutenberg, browndw/human-ai-parallel-corpus-mini (HAP-E), allenai/WildChat-1M, Jinyan1/COLING_2025_MGT_en, Yale-LILY/aeslc, text generated locally with google/gemma-3-1b-it, Photoroom/midjourney-v6-recap, OpenDatasets/dalle-3-dataset, ash12321/flux-1-dev-generated-10k and detection-datasets/coco. Training data is not redistributed.
- Runtime: [ONNX Runtime Web](https://onnxruntime.ai/) 1.30 (MIT). Example text on the page: Mark Twain, *Life on the Mississippi* (1883, public domain).

Code: [PolyForm Noncommercial 1.0.0](LICENSE.md), © 2026 Ritabrata Maiti. Model files: Apache-2.0 (text) and MIT (image); see [models/LICENSE-MODELS.txt](models/LICENSE-MODELS.txt).
