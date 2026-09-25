# Documentacion del proyecto

```
docs/
  papers/    articulos de referencia (PDF)
  informe/   el informe final (LaTeX, template NeurIPS 2025)
```

Las figuras y tablas del informe no viven aquí. Las genera
`python -m src.plots` en `results/figures/`: figuras en PDF (vectorial, para el
informe) y PNG (para el README), tablas en Markdown, PNG y LaTeX. Se
regeneran desde los JSON de `results/metrics/`, sin GPU.

## Referencias en `papers/`

- `BERT_paper_1810.04805.pdf`: Devlin et al. (2019), *BERT: Pre-training of
  Deep Bidirectional Transformers for Language Understanding*, NAACL.
- `ChangeTitans_Toward_Remote_Sensing_Change_Detection_With_Neural_Memory.pdf`:
  Yang et al. (2025), IEEE TGRS. La usamos como referencia de estilo para las
  figuras y las tablas.
