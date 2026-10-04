"""
Comando para lanzar destilacion (o fine-tuning supervisado) del
DistilBERT-custom implementado en `domain/`.

Ejemplos:

    # Destilacion real: profesor BERT -> estudiante DistilBERT-custom,
    # partiendo de los pesos pre-entrenados de distilbert-base-uncased.
    python -m src.custom_distilbert.cli.distill --task sst2 --epochs 2

    # Smoke test rapido (pocos ejemplos, para verificar que todo corre):
    python -m src.custom_distilbert.cli.distill --task sst2 --epochs 1 \\
        --max-train 64 --max-val 32 --batch-size 8 --log-every 2

    # Sin profesor (fine-tuning supervisado normal del modelo custom):
    python -m src.custom_distilbert.cli.distill --task sst2 --no-teacher

    # Arrancando desde pesos aleatorios (sin cargar distilbert-base-uncased):
    python -m src.custom_distilbert.cli.distill --task sst2 --no-pretrained
"""

import argparse
import json
import time
from pathlib import Path

from src.custom_distilbert.application.distillation_loss import DistillationWeights
from src.custom_distilbert.application.distill_trainer import (
    DistillTrainConfig,
    DistillTrainer,
)
from src.custom_distilbert.domain.config import DistilBertConfig
from src.custom_distilbert.domain.model import DistilBertForSequenceClassification
from src.custom_distilbert.infrastructure.hf_weight_loader import (
    load_pretrained_into_custom_model,
)
from src.custom_distilbert.infrastructure.teacher import Teacher
from src.data_adapter import load_task
from src.paths import RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="sst2", help="sst2 | ag_news | yelp")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--max-train", type=int, default=None,
                   help="Submuestrear train (recomendado para smoke test)")
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--log-every", type=int, default=20)

    p.add_argument("--no-pretrained", action="store_true",
                   help="No cargar pesos de distilbert-base-uncased: "
                        "arrancar el estudiante desde inicializacion aleatoria")
    p.add_argument("--no-teacher", action="store_true",
                   help="Entrenar solo con cross-entropy (sin destilacion)")
    p.add_argument("--teacher-hf-id", default="bert-base-uncased")
    p.add_argument("--teacher-checkpoint", default=None,
                   help="Ruta a un .pt de BERT ya afinado en la tarea. "
                        "Sin esto, el profesor no esta afinado (solo sirve "
                        "para probar que el pipeline corre, no para "
                        "resultados de destilacion reales)")

    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha-ce", type=float, default=1.0)
    p.add_argument("--beta-kd", type=float, default=1.0)
    p.add_argument("--gamma-cos", type=float, default=1.0)

    p.add_argument("--tag", default="custom-distill")
    p.add_argument("--out", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()

    ds, info_ds = load_task(
        args.task,
        subsample={k: v for k, v in
                  {"train": args.max_train, "validation": args.max_val}.items()
                  if v},
    )
    max_length = args.max_length or info_ds.max_length

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    config = DistilBertConfig(num_labels=info_ds.num_labels)
    student = DistilBertForSequenceClassification(config)

    if args.no_pretrained:
        print("[cli] estudiante con inicializacion ALEATORIA (--no-pretrained)")
    else:
        load_pretrained_into_custom_model(student, "distilbert-base-uncased")

    teacher = None
    if not args.no_teacher:
        teacher = Teacher(num_labels=info_ds.num_labels, hf_id=args.teacher_hf_id,
                          checkpoint_path=args.teacher_checkpoint)

    train_cfg = DistillTrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        log_every=args.log_every, use_teacher=not args.no_teacher,
        weights=DistillationWeights(
            alpha_ce=args.alpha_ce, beta_kd=args.beta_kd,
            gamma_cos=args.gamma_cos, temperature=args.temperature,
        ),
    )
    if teacher is not None:
        teacher.model.to(train_cfg.device)

    trainer = DistillTrainer(student, teacher, train_cfg)

    print("=" * 62)
    print(f"Destilacion custom | tarea={info_ds.name} | "
          f"profesor={'BERT' if teacher else 'ninguno (supervisado)'} | "
          f"dispositivo={train_cfg.device}")
    print("=" * 62)

    resultado = trainer.fit(ds, tokenizer, max_length)

    out_dir = Path(args.out) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / f"custom-distilbert_{info_ds.name}_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    destino.write_text(json.dumps(
        {"config": vars(args), "resultado": resultado}, indent=2, ensure_ascii=False,
        default=str,
    ), encoding="utf-8")
    print(f"\nResultados guardados en {destino}")


if __name__ == "__main__":
    main()
