# TinyDetect

AI-generated **text and image** detection that runs entirely in your browser. Nothing you paste or drop leaves your device.

**Live demo:** https://ritabratamaiti.github.io/tinydetect/

## What is different here

- **Goal-aware distillation.** Both students learn from a large teacher (soft labels) *and* from ground truth, with a loss that makes a false accusation (human work flagged as AI) cost 3x a miss, plus a hinge that pushes human scores below 0.3.
- **Custom text quantisation.** The vocabulary is pruned to the WordPiece tokens that actually occur (22,459 of 30,522), the embedding table is stored as **int4 with a per-row fp16 scale** and decoded in JavaScript, and only the transformer body runs in ONNX (dynamic int8). A ~60-line JS WordPiece tokenizer replaces the tokenizer library.
- **Image model**: MobileViT-xxs distilled from a Swin-B detector, quantised with a **custom weight-only per-channel int8** pass (int8 weights + fp32 scales, DequantizeLinear in-graph, activations kept fp32). Standard static int8 (QDQ) wrecked it: AUROC fell from 0.992 to 0.76 and 98% of real photos were flagged, because MobileViT's LayerNorm/attention activations do not survive 8-bit. Weight-only int8 loses nothing.

## Measured results

| Text (held-out, n=1,575) | Size | AUROC | Human text flagged as AI (p>0.5) |
|---|---|---|---|
| Teacher: RoBERTa-base detector | 499 MB | 0.872 | 72% |
| Student bert-mini, fp32 | ~44 MB | 0.971 | 10% |
| **Shipped** (pruned vocab + int4 emb + int8 body) | **6.86 MB** | **0.972** | **11%** |

| Image (held-out photos, n=400) | Size | AUROC | Real photos flagged as AI |
|---|---|---|---|
| Teacher: Swin-B detector | 347 MB | 0.892 | 33% |
| Student MobileViT-xxs, fp32 | ~5 MB | 0.992 | 4% |
| **Shipped** (custom weight-only int8) | **1.31 MB** | **0.992** | **4%** |
| Shipped, on a different source (low-res art thumbnails) | | 0.986 | 7% |

Text TPR at 1% FPR drops from 0.46 (fp32) to 0.33 (shipped): quantisation costs most at the strict end.

## Caveats (please read)

- The students were trained on data from the same sources they are tested on; the teachers were not. Part of the student gain is in-distribution, not magic.
- Paraphrasing, humanizer tools, new image generators, screenshots and edits will fool it. Article-deletion attacks drop text AUROC to ~0.85.
- **Do not use it to accuse anyone.** It is a hint, not evidence.

## Reproduce

The training folder has the full pipeline (data fetch, teacher scoring, distillation, quantisation, export). Trained on one RTX 4050 laptop GPU in minutes.

## Credits and licences

- Text teacher: [Oxidane/tmr-ai-text-detector](https://huggingface.co/Oxidane/tmr-ai-text-detector) (MIT). Text student init: [google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) (Apache-2.0).
- Image teacher: [Smogy/SMOGY-Ai-images-detector](https://huggingface.co/Smogy/SMOGY-Ai-images-detector) (CC BY-NC 4.0). Image student init: [apple/mobilevit-xx-small](https://huggingface.co/apple/mobilevit-xx-small) (Apple sample code licence). Also evaluated: [jacoballessio/ai-image-detect-distilled](https://huggingface.co/jacoballessio/ai-image-detect-distilled) (MIT).
- Data: [RAID](https://huggingface.co/datasets/liamdugan/raid) (MIT), [dmitva/human_ai_generated_text](https://huggingface.co/datasets/dmitva/human_ai_generated_text) (CC BY 4.0), eddyfox8812/ai-vs-real-2k-images and Hemg/AI-Generated-vs-Real-Images-Datasets (used for training/eval only, not redistributed).
- Runtime: [ONNX Runtime Web](https://onnxruntime.ai/) (MIT).

**Licence:** code under [PolyForm Noncommercial 1.0.0](LICENSE.md); model weights in models/ under CC BY-NC 4.0 (inherited from the image teacher). Copyright 2026 Ritabrata Maiti.