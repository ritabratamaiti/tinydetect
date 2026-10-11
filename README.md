# TinyDetect

AI text and image detection that runs in your browser. The models are downloaded once and run locally with ONNX Runtime Web (WebGPU when available, WebAssembly otherwise), so nothing you check is sent anywhere.

**Demo:** https://ritabratamaiti.github.io/tinydetect/

## How it works

**Text.** The detector uses Fakespot RoBERTa with a blend of original and lightly tuned weights (WiSE-FT). Word embeddings are stored separately in int4 and decoded in JavaScript; the transformer uses mixed-precision quantisation. A byte-level BPE tokenizer and Fakespot's `clean_text` preprocessing run in JavaScript.

**Image.** The detector uses a quantised Community Forensics vision transformer with calibrated output scores. Images are resized, centre-cropped and normalised locally before inference.

**Runtime.** `boot.js` selects ONNX Runtime's WebGPU build when a GPU adapter is available and its WebAssembly build otherwise. A service worker adds cross-origin isolation headers for threaded WebAssembly on GitHub Pages and caches model and runtime files. A warm-up run compiles GPU shaders before the first check.

**Limitations.** Results are signals, not proof of authorship. Short or casual text, paraphrasing, heavy editing, screenshots, cropped images and new generators can fool both models. Do not use a score alone to make decisions about someone's grade or job.

## Setup

Clone the repository and serve its root with a local HTTP server, for example:

```sh
git clone https://github.com/ritabratamaiti/tinydetect.git
cd tinydetect
python -m http.server 8000
```

Open `http://localhost:8000` in your browser. Keep `models/`, `ort/` and `ort_gpu/` alongside the page and scripts. The first visit downloads the models; subsequent visits can reuse cached files. No backend or API key is required.

For GitHub Pages, publish the repository root from `main` and retain `.nojekyll`.

## Reproduce

`training/` has the scripts: candidate benchmarks (`bench_candidates.py`), text tuning (`tune_fakespot.py`, `wise_fakespot.py`, `pick_alpha.py`, `make_wise.py`), export and quantisation (`export_text.py`, `mixed_text.py`, `mix_apply.py`, `optimise_image.py`, `quant_sweep_image.py`) and evaluation (`eval_text_variants.py`). Everything runs on one laptop.

## Credits and licences

- Text model: [fakespot-ai/roberta-base-ai-text-detection-v1](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1) (Apache-2.0), based on [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base). The preprocessing in `app.js` is ported from its `utils.py`.
- Image model: [OwensLab/commfor-model-384](https://huggingface.co/OwensLab/commfor-model-384), [Community Forensics](https://github.com/JeongsooP/Community-Forensics) (MIT, © 2025 Jeongsoo Park).
- Tuning and calibration data: Hello-SimpleAI/HC3, aadityaubhat/GPT-wiki-intro, Project Gutenberg, browndw/human-ai-parallel-corpus-mini (HAP-E), allenai/WildChat-1M, Jinyan1/COLING_2025_MGT_en, Yale-LILY/aeslc, text generated locally with google/gemma-3-1b-it, Photoroom/midjourney-v6-recap, OpenDatasets/dalle-3-dataset, ash12321/flux-1-dev-generated-10k and detection-datasets/coco. Training data is not redistributed.
- Runtime: [ONNX Runtime Web](https://onnxruntime.ai/) 1.30 (MIT). Example text on the page: Mark Twain, *Life on the Mississippi* (1883, public domain).

Code: [PolyForm Noncommercial 1.0.0](LICENSE.md), © 2026 Ritabrata Maiti. Model files: Apache-2.0 (text) and MIT (image); see [models/LICENSE-MODELS.txt](models/LICENSE-MODELS.txt).
