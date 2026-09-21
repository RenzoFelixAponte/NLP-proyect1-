#!/bin/bash
# ---------------------------------------------------------------------------
# Ablation study de la cabeza de clasificacion (solo DistilBERT).
#
#   sbatch scripts/ablation.sbatch                 # 4 configuraciones x 3 datasets
#   CONFIGS=6 sbatch scripts/ablation.sbatch       # 6 configuraciones x 3 datasets
#   TAREAS="sst2" sbatch scripts/ablation.sbatch   # solo un dataset
#   MODELO=bert CONFIGS=mejor sbatch scripts/ablation.sbatch   # la mejor, con BERT
#
# Es UN solo job que ejecuta los experimentos uno detras de otro, en vez de
# un job por experimento. La QOS 'a-pregrado' solo admite 3 jobs encolados y
# una GPU a la vez, asi que de todos modos correrian en serie: agruparlos
# evita chocar con ese limite y ahorra el tiempo de cola de cada uno.
#
# Todas las configuraciones comparten dataset, semilla, optimizador, epocas y
# batch. Lo unico que cambia es la cabeza y que parte del transformer se
# entrena, que es justo lo que el estudio quiere medir.
# ---------------------------------------------------------------------------
#SBATCH --job-name=nlp-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.out

set -uo pipefail

CONDA_ENV="${CONDA_ENV:-nlp-p1}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MODELO="${MODELO:-distilbert}"
TAREAS="${TAREAS:-sst2 ag_news yelp}"
CONFIGS="${CONFIGS:-4}"

# --- Configuraciones -------------------------------------------------------
# Formato: "etiqueta|argumentos". Los tres ejes que pide el enunciado son
# el numero de capas de la cabeza, el numero de neuronas por capa, y que
# parte del transformer se congela.
case "$CONFIGS" in
  4)
    # Minimo del enunciado: una referencia y una variacion por cada eje.
    LISTA=(
      "h768|--head-hidden 768"                              # referencia: 1 capa de 768
      "c2|--head-hidden 512,256"                            # mas capas: 2 ocultas
      "h2048|--head-hidden 2048"                            # mas neuronas
      "frz|--head-hidden 768 --freeze-encoder --lr 1e-3"    # transformer congelado
    )
    ;;
  6)
    # Cubre los dos sentidos de cada eje: mas y menos capas, mas y menos
    # neuronas. Con 4 solo se ve una direccion y no se puede afirmar si la
    # metrica sube o baja de forma monotona.
    LISTA=(
      "lineal|--head-hidden none"                           # menos capas: 0 ocultas
      "h128|--head-hidden 128"                              # menos neuronas
      "h768|--head-hidden 768"                              # referencia
      "h2048|--head-hidden 2048"                            # mas neuronas
      "c2|--head-hidden 512,256"                            # mas capas
      "frz|--head-hidden 768 --freeze-encoder --lr 1e-3"    # transformer congelado
    )
    ;;
  mejor)
    # Se rellena a mano con la ganadora del ablation, para repetirla con BERT.
    LISTA=( "${MEJOR:-h768|--head-hidden 768}" )
    ;;
  *)
    echo "CONFIGS debe ser 4, 6 o 'mejor'"; exit 1
    ;;
esac

echo "=== job $SLURM_JOB_ID en $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== modelo=$MODELO tareas='$TAREAS' configs=$CONFIGS (${#LISTA[@]}) ==="

# El congelado usa lr 1e-3 en lugar de 2e-5. Ese lr esta pensado para ajustar
# pesos ya pre-entrenados; la cabeza de un encoder congelado se entrena desde
# cero y con 2e-5 apenas avanzaria: perderia por el learning rate y no por
# estar congelada, que es lo que se quiere medir.

fallos=0
for tarea in $TAREAS; do
    # Yelp se submuestrea igual que en las corridas base, para que el
    # ablation se lea contra ellas.
    extra=""
    [ "$tarea" = "yelp" ] && extra="--max-train 100000 --max-val 20000"

    for entrada in "${LISTA[@]}"; do
        tag="${entrada%%|*}"
        args="${entrada#*|}"
        echo ""
        echo "############ $MODELO | $tarea | abl-$tag ############"
        # shellcheck disable=SC2086
        srun python -m src.train --model "$MODELO" --task "$tarea" \
            --epochs 2 --batch-size 32 --log-every 100 --eval-every 500 \
            --tag "abl-$tag" $extra $args || { echo "FALLO: $tarea/$tag"; fallos=$((fallos+1)); }
    done
done

echo ""
echo "=== terminado: ${#LISTA[@]} configuraciones x $(echo $TAREAS | wc -w) datasets, $fallos fallos ==="
exit $fallos
