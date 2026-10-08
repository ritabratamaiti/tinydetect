# TinyDetect

AI text and image detection that runs in your browser. The models are downloaded once and run locally with ONNX Runtime Web, so nothing you check is sent anywhere.

**Demo:** https://ritabratamaiti.github.io/tinydetect/

| | Size | Distilled from |
|---|---|---|
| Text | 7.41 MB | RoBERTa-base detector, 499 MB |
| Image | 1.31 MB | Swin-B detector, 347 MB |

## How it works

**Distillation with a cost on false accusations.** Each student learns from its teacher's scores and from the true labels. Mistaking human work for AI costs three times as much as missing AI work, and a hinge term pushes human scores below 0.3. The text student only copies the teacher on AI samples, because the teacher flags a lot of formal human writing (it flagged 48% of held-out Wikipedia/news/abstract/review text; the student flags 8%).

**Text quantisation.** The WordPiece vocabulary is cut to the tokens that occur in the training data. The embedding table is stored as int4 with a per-row fp16 scale and decoded in JavaScript. The transformer body runs in ONNX with dynamic int8. A short WordPiece tokenizer in `app.js` replaces the tokenizer library; it also strips zero-width characters.

**Image quantisation.** Standard static int8 (QDQ) broke the MobileViT student: AUROC fell from 0.992 to 0.76 and 98% of real photos were flagged, because its LayerNorm and attention activations don't survive 8 bits. A custom weight-only pass (int8 weights, fp32 scales, DequantizeLinear in the graph, fp32 activations) loses nothing measurable and takes the file from 4.06 MB to 1.31 MB.

## Benchmarks

**Fresh samples, set 2.** Collected after the models were finished and never used to tune them. Human text: Hacker News comments from 2014, Project Gutenberg novels, Wikipedia as of 2019. AI text: 16 passages written by Claude, most of them deliberately casual (short emails, a Slack message, Reddit advice). Real photos: Wikimedia Commons uploads from before 2019. AI images: Midjourney, DALL·E 3 and Flux, from sources not used in training.

| | n | AI-generated (≥0.9) | Possibly AI (0.6–0.9) | Unclear (0.3–0.6) | Human (<0.3) |
|---|---|---|---|---|---|
| Human writing | 24 | 1 | 0 | 0 | 23 |
| AI writing | 16 | 4 | 2 | 1 | 9 |
| Real photos | 21 | 0 | 0 | 2 | 19 |
| AI images | 31 | 20 | 5 | 3 | 3 |

**Fresh samples, set 1.** Hacker News comments from 2012, Project Gutenberg, Wikipedia as of 2019; 16 Claude passages in mixed styles; Wikimedia Commons photos; mostly Midjourney images. This set was used to choose the score bands and to find failure cases, and some of its DALL·E 3 and Midjourney v6 images come from datasets that later went into training (different images), so it is less clean than set 2.

| | n | AI-generated | Possibly AI | Unclear | Human |
|---|---|---|---|---|---|
| Human writing | 24 | 3 | 1 | 1 | 19 |
| AI writing | 16 | 11 | 0 | 1 | 4 |
| Real photos | 30 | 0 | 1 | 1 | 28 |
| AI images | 50 | 28 | 8 | 6 | 8 |

Per-sample scores are in `training/fresh_results.json` and `training/fresh2_results.json`.

**What it misses, and why.**

- *Casual AI text.* Short emails, chat messages and forum replies written to sound like a person mostly come back as human. The 499 MB teacher does flag them, but it also flags most real Hacker News comments, so it isn't telling them apart either. I tried adding casual AI text from three public datasets (COLING-2025 MGT, WildChat, HAP-E) and from a local Gemma 3 1B model. None of it improved set 2 without raising false flags on real writing, so the shipped text model is the one that accuses real people least.
- *Formal human prose.* A few 2019 Wikipedia paragraphs score as AI. Earlier versions were much worse because almost all the book and abstract text in the training data was AI-written; capping AI samples per source and adding paired human/AI text from the same sources (HC3, GPT-wiki-intro) and more Gutenberg fixed most of it.
- *Illustrated Midjourney art.* The first image model had never seen Midjourney-style art and flagged real photos too often. Retraining with Midjourney v6, DALL·E 3 and Flux images, balanced with COCO photos, cut real photos called AI from 7 of 51 to 1 of 51 across both sets, and caught one more AI image (61 of 81). Painterly Midjourney images are still the main miss.

**Held-out test sets.**

| Text, n=1,745, 17 sources | Size | AUROC | Human text flagged (>0.5) |
|---|---|---|---|
| RoBERTa teacher | 499 MB | 0.955 | 44% |
| TinyDetect | 7.41 MB | 0.979 | 7% |

| Images, n=400 | Size | AUROC | Real photos flagged (>0.5) |
|---|---|---|---|
| Swin-B teacher | 347 MB | 0.892 | 33% |
| TinyDetect | 1.31 MB | 0.984 | 5% |

The students were trained on the same sources as these held-out sets and the teachers were not, so the fresh samples above are the fairer comparison.

## Reproduce

`training/` has the data fetchers, distillation, quantisation and evaluation scripts. Everything trains on one laptop GPU (RTX 4050) in a few minutes.

## Credits and licences

- Text: teacher [Oxidane/tmr-ai-text-detector](https://huggingface.co/Oxidane/tmr-ai-text-detector) (MIT); student initialised from [google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) (Apache-2.0).
- Image: teacher [Smogy/SMOGY-Ai-images-detector](https://huggingface.co/Smogy/SMOGY-Ai-images-detector) (CC BY-NC 4.0); student initialised from [apple/mobilevit-xx-small](https://huggingface.co/apple/mobilevit-xx-small) (Apple Sample Code License).
- Data: [RAID](https://huggingface.co/datasets/liamdugan/raid) (MIT), [dmitva/human_ai_generated_text](https://huggingface.co/datasets/dmitva/human_ai_generated_text) (CC BY 4.0), Salesforce/wikitext, vblagoje/cc_news, gfissore/arxiv-abstracts-2021, Yelp/yelp_review_full, webis/tldr-17, Hello-SimpleAI/HC3, aadityaubhat/GPT-wiki-intro, Project Gutenberg, eddyfox8812/ai-vs-real-2k-images and Hemg/AI-Generated-vs-Real-Images-Datasets, Photoroom/midjourney-v6-recap, OpenDatasets/dalle-3-dataset, ash12321/flux-1-dev-generated-10k and detection-datasets/coco. Training data is not redistributed.
- Runtime: [ONNX Runtime Web](https://onnxruntime.ai/) (MIT). Example text on the page: Mark Twain, *Life on the Mississippi* (1883, public domain).

Code: [PolyForm Noncommercial 1.0.0](LICENSE.md). Model weights: CC BY-NC 4.0 (inherited from the image teacher). © 2026 Ritabrata Maiti.
