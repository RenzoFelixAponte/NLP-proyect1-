Ablation sobre distilbert. La fila resaltada es la configuracion elegida (mayor F1 medio de validacion).

| Config | Head | Encoder | Validation F1 (%): SST-2 | Validation F1 (%): AG News | Validation F1 (%): Yelp | Validation F1 (%): Mean | Trainable (M) | Train (min) |
|---|---|---|---|---|---|---|---|---|
| lineal | 768 → C | trained | 94.83 | **94.62** | 96.03 | **95.16** | 66.4 | 15.3 |
| h128 | 768 → 128 → C | trained | 94.80 | 94.54 | 95.93 | 95.09 | 66.5 | 15.9 |
| h768 | 768 → 768 → C | trained | **94.85** | 94.27 | 95.96 | 95.03 | 67.0 | 16.3 |
| h2048 | 768 → 2048 → C | trained | 94.70 | 94.59 | 96.01 | 95.10 | 67.9 | 16.3 |
| c2 | 768 → 512 → 256 → C | trained | 94.76 | 94.34 | **96.04** | 95.05 | 66.9 | 15.9 |
| frz | 768 → 768 → C | frozen | 85.74 | 90.84 | 90.05 | 88.87 | **0.6** | **6.0** |
