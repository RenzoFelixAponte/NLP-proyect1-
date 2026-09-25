Configuracion `lineal` (linear) en los dos modelos, misma receta. En negrita, el mejor de cada dataset.

| Dataset | Model | Performance (%): Acc | Performance (%): Prec | Performance (%): Rec | Performance (%): F1 | Performance (%): Val F1 | Efficiency: Params (M) | Efficiency: Latency (ms) | Efficiency: Memory (MB) | Efficiency: Train (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| SST-2 | BERT | **92.89** | **92.94** | **92.86** | **92.88** | **95.41** | 109.5 | 5.56 | 430 | 4.2 |
| SST-2 | DistilBERT | 90.94 | 90.99 | 90.91 | 90.93 | 94.83 | **66.4** | **2.60** | **269** | **1.7** |
| AG News | BERT | **94.53** | **94.56** | **94.53** | **94.53** | **94.81** | 109.5 | 5.61 | 433 | 9.0 |
| AG News | DistilBERT | 94.42 | 94.43 | 94.42 | 94.42 | 94.62 | **66.4** | **2.61** | **272** | **5.4** |
| Yelp | BERT | **96.53** | **96.53** | **96.53** | **96.53** | **96.61** | 109.5 | 5.11 | 436 | 16.0 |
| Yelp | DistilBERT | 96.10 | 96.10 | 96.10 | 96.10 | 96.03 | **66.4** | **2.74** | **275** | **8.2** |
