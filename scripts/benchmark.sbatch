#!/bin/bash
# ---------------------------------------------------------------------------
# Mide la eficiencia de los modelos en un nodo GPU (src/benchmark.py).
#
# Todo lo que se le pase se reenvia tal cual a src/benchmark.py:
#
#   sbatch scripts/benchmark.sbatch
#   sbatch scripts/benchmark.sbatch --models bert distilbert --batch-sizes 1 32
#
# Mide los dos modelos en la MISMA rejilla de (batch, longitud), en la misma
# GPU y en el mismo proceso. Es lo que hace comparables los numeros: la
# latencia que guarda cada fine-tuning se mide con la longitud de SU dataset
# y no se puede poner en la misma tabla.
#
# Los nodos de computo NO tienen salida a internet: los datasets y los
# checkpoints se descargan antes desde el nodo de login (ver README) y aqui
# se trabaja en modo offline. Si falta algo en la cache, el job falla rapido
# en vez de quedarse colgado esperando a HuggingFace.
# ---------------------------------------------------------------------------
#SBATCH --job-name=nlp-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.out

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-nlp-p1}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Sin internet en el nodo: se usa exclusivamente la cache local.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# El tokenizador rapido avisa de un posible deadlock al combinarse con los
# workers del DataLoader; aqui se tokeniza en el proceso principal, asi que
# se desactiva el paralelismo y se evita el ruido en el log.
export TOKENIZERS_PARALLELISM=false

echo "=== job $SLURM_JOB_ID en $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=== args: $* ==="

srun python -m src.benchmark "$@"
