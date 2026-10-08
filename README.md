# TinyDetect

AI text and image detection that runs in your browser. The models are downloaded once and run locally with ONNX Runtime Web, so nothing you check is sent anywhere.

**Demo:** https://ritabratamaiti.github.io/tinydetect/

| | Size | Distilled from |
|---|---|---|
| Text | 7.28 MB | RoBERTa-base detector, 499 MB |
| Image | 1.31 MB | Swin-B detector, 347 MB |

## How it works

**Distillation with a cost on false accusations.** Each student learns from its teacher's scores and from the true labels. Mistaking human work for AI costs three times as much as missing AI work, and a hinge term pushes human scores below 0.3. The text student only copies the teacher on AI samples, because the teacher flags a lot of formal human writing (it flagged 43% of held-out Wikipedia/news/abstract/review text; the student flags 13%).

**Text quantisation.** The WordPiece vocabulary is cut to the 25,305 tokens that occur in the training data (of 30,522). The embedding table is stored as int4 with a per-row fp16 scale and decoded in JavaScript. The transformer body runs in ONNX with dynamic int8. A short WordPiece tokenizer in `app.js` replaces the tokenizer library; it also strips zero-width characters.

**Image quantisation.** Standard static int8 (QDQ) broke the MobileViT student: AUROC fell from 0.992 to 0.76 and 98% of real photos were flagged, because its LayerNorm and attention activations don't survive 8 bits. A custom weight-only pass (int8 weights, fp32 scales, DequantizeLinear in the graph, fp32 activations) loses nothing and takes the file from 4.06 MB to 1.31 MB.

## Benchmarks

**Fresh samples** (none seen in training). Human text: Hacker News comments from 2012, Project Gutenberg, Wikipedia as of 2019. AI text: 16 passages written by Claude. Real photos: Wikimedia Commons uploads before 2019. AI images: Midjourney and DALL·E 3.

| | n | AI-generated (≥0.9) | Possibly AI (0.6–0.9) | Unclear (0.3–0.6) | Human (<0.3) |
|---|---|---|---|---|---|
| Human writing | 24 | 2 | 5 | 4 | 13 |
| AI writing | 16 | 13 | 0 | 0 | 3 |
| Real photos | 30 | 2 | 1 | 0 | 27 |
| AI images | 50 | 33 | 1 | 6 | 10 |

The AI text it misses is very casual (Reddit-style posts, a short email). Per-sample scores are in `training/fresh_results.json`.

**Held-out test sets.**

| Text, RAID, n=1,935, nine genres | Size | AUROC | Human text flagged (>0.5) |
|---|---|---|---|
| RoBERTa teacher | 499 MB | 0.905 | 54% |
| TinyDetect | 7.28 MB | 0.980 | 11% |

| Images, n=400 photos | Size | AUROC | Real photos flagged (>0.5) |
|---|---|---|---|
| Swin-B teacher | 347 MB | 0.892 | 33% |
| TinyDetect | 1.31 MB | 0.992 | 4% |

The students were trained on the same sources as these held-out sets and the teachers were not, so the fresh samples above are the fairer comparison.

## Reproduce

`training/` has the data fetchers, distillation, quantisation and evaluation scripts. Everything trains on one laptop GPU (RTX 4050) in a few minutes.

## Credits and licences

- Text: teacher [Oxidane/tmr-ai-text-detector](https://huggingface.co/Oxidane/tmr-ai-text-detector) (MIT); student initialised from [google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) (Apache-2.0).
- Image: teacher [Smogy/SMOGY-Ai-images-detector](https://huggingface.co/Smogy/SMOGY-Ai-images-detector) (CC BY-NC 4.0); student initialised from [apple/mobilevit-xx-small](https://huggingface.co/apple/mobilevit-xx-small) (Apple Sample Code License).
- Data: [RAID](https://huggingface.co/datasets/liamdugan/raid) (MIT), [dmitva/human_ai_generated_text](https://huggingface.co/datasets/dmitva/human_ai_generated_text) (CC BY 4.0), Salesforce/wikitext, vblagoje/cc_news, gfissore/arxiv-abstracts-2021, Yelp/yelp_review_full, webis/tldr-17, eddyfox8812/ai-vs-real-2k-images and Hemg/AI-Generated-vs-Real-Images-Datasets. Training data is not redistributed.
- Runtime: [ONNX Runtime Web](https://onnxruntime.ai/) (MIT). Example text on the page: Mark Twain, *Life on the Mississippi* (1883, public domain).

Code: [PolyForm Noncommercial 1.0.0](LICENSE.md). Model weights: CC BY-NC 4.0 (inherited from the image teacher). © 2026 Ritabrata Maiti.
