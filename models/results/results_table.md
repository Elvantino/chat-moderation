| Tier | Model | Technique Area | Precision | Macro-F1 | Size (MB) | Avg Latency (ms, CPU, n=200) | p95 (ms) | < 50 ms |
|---|---|---|---|---:|---:|---:|---:|:---:|
| Rule baseline | Lexicon rule (edit distance) | Area B – Classical NLP | rules | 0.7777 | 0.002 | 0.71 | 2.11 | ✅ |
| Baseline | TF-IDF + LogReg | Area B – Classical NLP | fp64 | 0.8925 | 4.33 | 2.84 | 5.33 | ✅ |
| Neural Net | Word2Vec + FFNN (SGD) | Area A – Neural Networks | fp32 | 0.8635 | 34.53 | 0.31 | 0.73 | ✅ |
| Transformer | BERT-base (fp32) | Area C – Transformers | fp32 | 0.8606 | 418.57 | 84.57 | 136.51 | ❌ |
| Quantized | BERT-base (dynamic INT8) | Area D – Efficient Models | int8 (Linear) + fp32 (embeddings) | 0.8521 | 173.99 | 38.67 | 74.42 | ✅ |

_Macro-F1 on the same 15,958-message test split. Latency: batch size 1, mean of 200 runs after warm-up, CPU only. Hardware: Intel(R) Xeon(R) Processor @ 2.30GHz (2 vCPUs, cloud VM), 7 GB RAM, CPU only._
