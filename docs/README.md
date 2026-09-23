# Documentacion del proyecto

```
docs/
  papers/    articulos de referencia (PDF)
  informe/   el informe final y lo que se use para escribirlo
```

Las figuras y tablas del informe NO viven aqui: se generan con
`python -m src.plots` y salen a `results/figures/` en PDF (para incrustar en
el informe, que es vectorial) y en PNG (para el README, que GitHub sí
renderiza). Se regeneran desde los JSON de `results/metrics/`, sin GPU.

## Referencias en `papers/`

- `BERT_paper_1810.04805.pdf` — Devlin et al. (2019), *BERT: Pre-training of
  Deep Bidirectional Transformers for Language Understanding*, NAACL.
