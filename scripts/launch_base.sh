#!/bin/bash
# ---------------------------------------------------------------------------
# Fine-tuning base de UN modelo sobre los tres datasets del proyecto.
#
#   bash scripts/launch_base.sh distilbert
#   bash scripts/launch_base.sh bert
#
# Los tres jobs se envian al mismo nodo (g002, RTX A6000) a proposito: las
# metricas de eficiencia (latencia y memoria GPU) solo son comparables entre
# si cuando se miden en el mismo hardware. Como el nodo tiene una sola GPU,
# Slurm los ejecuta uno detras de otro.
#
# Presupuesto de entrenamiento (igual para todos los modelos):
#   SST-2   : train completo (60.6k), len 64
#   AG News : train completo (108k),  len 128
#   Yelp    : 100k de 560k,           len 256  <- submuestreado por tiempo
# 2 epocas, batch 32, lr 2e-5, warmup 10%.
# ---------------------------------------------------------------------------
set -euo pipefail

MODELO="${1:-distilbert}"
NODO="${NODO:-g002}"
TAG="${TAG:-base}"

comun=(--model "$MODELO" --epochs 2 --batch-size 32 --lr 2e-5 --tag "$TAG")

sbatch -w "$NODO" --job-name="${MODELO}-sst2" scripts/train.sbatch \
    "${comun[@]}" --task sst2 --log-every 100 --eval-every 500

sbatch -w "$NODO" --job-name="${MODELO}-agnews" scripts/train.sbatch \
    "${comun[@]}" --task ag_news --log-every 100 --eval-every 500

# En Yelp tambien se submuestrea la validacion: el 10% del train son 56k
# resenas y evaluarlas cada 500 pasos costaria mas que el propio entrenamiento.
sbatch -w "$NODO" --job-name="${MODELO}-yelp" scripts/train.sbatch \
    "${comun[@]}" --task yelp --max-train 100000 --max-val 20000 \
    --log-every 100 --eval-every 500

squeue -u "$USER"
