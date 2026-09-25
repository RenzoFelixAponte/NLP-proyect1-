Corridas base. Precision, recall y F1 son macro. Latencia y memoria: benchmark controlado a batch 1 con la longitud del dataset. En negrita, el mejor de cada dataset.

| Dataset | Model | Test (%): Acc | Test (%): Prec | Test (%): Rec | Test (%): F1 | Efficiency: Params (M) | Efficiency: Latency (ms) | Efficiency: Memory (MB) | Efficiency: Train (min) |
|---|---|---|---|---|---|---|---|---|---|
| SST-2 | BERT | **92.32** | **92.32** | **92.31** | **92.31** | 109.5 | 5.56 | 430 | 4.2 |
| SST-2 | DistilBERT | 91.28 | 91.34 | 91.25 | 91.27 | **67.0** | **2.60** | **269** | **2.7** |
| AG News | BERT | 94.41 | 94.43 | 94.41 | 94.41 | 109.5 | 5.61 | 433 | 9.1 |
| AG News | DistilBERT | **94.50** | **94.52** | **94.50** | **94.50** | **67.0** | **2.61** | **272** | **5.5** |
| Yelp | BERT | **96.52** | **96.52** | **96.52** | **96.52** | 109.5 | 5.11 | 436 | 16.0 |
| Yelp | DistilBERT | 96.06 | 96.06 | 96.06 | 96.06 | **67.0** | **2.74** | **275** | **8.2** |
