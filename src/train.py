"""
Fine-tuning de BERT (implementacion propia) para clasificacion de texto.

Bucle de entrenamiento escrito a mano: no se usa el Trainer de HuggingFace.
De esa libreria solo se toma el tokenizador WordPiece, porque el vocabulario
es parte del checkpoint pre-entrenado (usar otro romperia la correspondencia
token <-> embedding).

Uso:
    python -m src.train --task sst2 --epochs 2
    python -m src.train --task sst2 --max-train 2000 --epochs 1   # prueba rapida
"""

import os

# Anaconda y PyTorch traen cada uno su runtime de OpenMP y chocan en Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# IMPORTANTE: torch debe importarse ANTES que numpy. En este entorno
# (Anaconda + torch 2.5.1 en Windows), si numpy carga primero, torch falla
# con "Error loading fbgemm.dll" por un conflicto de DLLs.
import torch
from torch.utils.data import DataLoader, TensorDataset

import json
import time
import argparse
from pathlib import Path

import numpy as np

from bert import BertConfig, BertForSequenceClassification, load_pretrained, count_parameters
from src.data_adapter import load_task
from src.metrics import (
    classification_metrics,
    per_class_report,
    confusion,
    measure_latency,
    measure_gpu_memory,
    model_size_mb,
)

MODEL_ID = "bert-base-uncased"
SEED = 42


def set_seed(seed=SEED):
    """Fija todas las fuentes de aleatoriedad para que el run sea reproducible."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(split, tokenizer, max_length, batch_size, shuffle):
    """
    Tokeniza un split completo y lo envuelve en un DataLoader.

    Se tokeniza todo de golpe con padding a max_length fijo. Para datasets
    de este tamano cabe en RAM sin problema y evita re-tokenizar en cada
    epoca.
    """
    # list(...) es necesario: en datasets 5.x, split["text"] devuelve un
    # objeto Column, no una lista, y el tokenizador lo rechaza.
    enc = tokenizer(
        list(split["text"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(list(split["label"]), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Evalua el modelo: devuelve loss media, predicciones y etiquetas."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            loss, logits = model(input_ids, attention_mask=attention_mask,
                                 labels=labels)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    return (total_loss / max(n_batches, 1),
            torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sst2", help="sst2 | ag_news | yelp")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="El paper recomienda 5e-5, 3e-5 o 2e-5 para fine-tuning")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Submuestrear el train (pruebas rapidas o Yelp)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Por defecto, el recomendado por el dataset")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Cada cuantos pasos registrar la loss")
    parser.add_argument("--eval-every", type=int, default=200,
                        help="Cada cuantos pasos evaluar en validacion")
    parser.add_argument("--no-amp", action="store_true",
                        help="Desactivar precision mixta fp16")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    print("=" * 62)
    print(f"Fine-tuning BERT-base (implementacion propia) | tarea: {args.task}")
    print("=" * 62)
    print(f"Dispositivo : {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
    print(f"Precision   : {'fp16 (AMP)' if use_amp else 'fp32'}")

    # --- Datos -------------------------------------------------------------
    subsample = {"train": args.max_train} if args.max_train else None
    ds, info = load_task(args.task, subsample=subsample)
    max_length = args.max_length or info.max_length

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"\nDataset     : {info.name} ({info.num_labels} clases)")
    print(f"  train / val / test : {info.n_train:,} / {info.n_val:,} / {info.n_test:,}")
    print(f"  max_length         : {max_length}")

    train_loader = build_dataloader(ds["train"], tokenizer, max_length,
                                    args.batch_size, shuffle=True)
    val_loader = build_dataloader(ds["validation"], tokenizer, max_length,
                                  args.batch_size, shuffle=False)
    test_loader = build_dataloader(ds["test"], tokenizer, max_length,
                                   args.batch_size, shuffle=False)

    # --- Modelo ------------------------------------------------------------
    config = BertConfig.base()
    model = BertForSequenceClassification(config, num_labels=info.num_labels)
    load_pretrained(model, verbose=False)
    model.to(device)

    n_params = count_parameters(model)
    print(f"\nModelo      : {n_params:,} parametros ({model_size_mb(model):.1f} MB en fp32)")

    # --- Optimizador -------------------------------------------------------
    # No se aplica weight decay a bias ni a los parametros de LayerNorm:
    # son terminos de escala/desplazamiento, penalizarlos degrada el modelo.
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr, eps=1e-8)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        """Warmup lineal y luego decaimiento lineal, como en el paper."""
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Optimizador : AdamW lr={args.lr}, warmup {warmup_steps}/{total_steps} pasos")

    # --- Entrenamiento -----------------------------------------------------
    historia = []          # para el grafico "iteraciones vs loss"
    paso_global = 0
    mejor_f1 = -1.0
    mejor_estado = None
    t0 = time.time()

    for epoca in range(args.epochs):
        print(f"\n--- Epoca {epoca + 1}/{args.epochs} ---")
        model.train()
        acum_loss, acum_n = 0.0, 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                loss, _ = model(input_ids, attention_mask=attention_mask,
                                labels=labels)

            scaler.scale(loss).backward()
            # Hay que des-escalar antes de recortar, o el umbral se aplica
            # sobre gradientes escalados y no significa lo que se cree.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acum_loss += loss.item()
            acum_n += 1
            paso_global += 1

            if paso_global % args.log_every == 0:
                train_loss = acum_loss / acum_n
                acum_loss, acum_n = 0.0, 0
                registro = {"paso": paso_global, "train_loss": train_loss,
                            "val_loss": None, "lr": scheduler.get_last_lr()[0]}

                if paso_global % args.eval_every == 0:
                    val_loss, val_pred, val_true = evaluate(model, val_loader,
                                                            device, use_amp)
                    val_acc = classification_metrics(val_true, val_pred)["accuracy"]
                    registro["val_loss"] = val_loss
                    registro["val_accuracy"] = val_acc
                    print(f"  paso {paso_global:5d} | train_loss {train_loss:.4f} "
                          f"| val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")
                    model.train()
                else:
                    print(f"  paso {paso_global:5d} | train_loss {train_loss:.4f}")

                historia.append(registro)

        # Evaluacion al cierre de cada epoca
        val_loss, val_pred, val_true = evaluate(model, val_loader, device, use_amp)
        m = classification_metrics(val_true, val_pred)
        print(f"  [fin epoca {epoca+1}] val_loss {val_loss:.4f} | "
              f"acc {m['accuracy']:.4f} | f1 {m['f1']:.4f}")

        if m["f1"] > mejor_f1:
            mejor_f1 = m["f1"]
            mejor_estado = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            print(f"  -> mejor modelo hasta ahora (f1 = {mejor_f1:.4f})")

    tiempo_entrenamiento = time.time() - t0
    print(f"\nEntrenamiento completado en {tiempo_entrenamiento/60:.1f} min")

    # --- Evaluacion final en TEST con el mejor checkpoint -----------------
    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)

    test_loss, test_pred, test_true = evaluate(model, test_loader, device, use_amp)
    desempeno = classification_metrics(test_true, test_pred)

    print("\n" + "=" * 62)
    print(f"RESULTADOS EN TEST ({info.name})")
    print("=" * 62)
    print(f"  Accuracy  : {desempeno['accuracy']:.4f}")
    print(f"  Precision : {desempeno['precision']:.4f}")
    print(f"  Recall    : {desempeno['recall']:.4f}")
    print(f"  F1-score  : {desempeno['f1']:.4f}")

    print("\n  Por clase:")
    for fila in per_class_report(test_true, test_pred, info.class_names):
        print(f"    {fila['clase']:10s} P {fila['precision']:.3f}  "
              f"R {fila['recall']:.3f}  F1 {fila['f1']:.3f}  (n={fila['n']})")

    # --- Metricas de eficiencia -------------------------------------------
    print("\n--- Eficiencia ---")
    ejemplo = next(iter(test_loader))
    # Latencia con batch=1: es la medida que importa para servir en produccion
    lat = measure_latency(model, ejemplo[0][:1], ejemplo[1][:1], device)
    mem = measure_gpu_memory(model, ejemplo[0][:1], ejemplo[1][:1], device)
    print(f"  Parametros        : {n_params:,}")
    print(f"  Tamano (fp32)     : {model_size_mb(model):.1f} MB")
    print(f"  Latencia (batch=1): {lat['latencia_ms_media']:.2f} +/- "
          f"{lat['latencia_ms_std']:.2f} ms")
    if mem.get("memoria_mb_pico"):
        print(f"  Memoria GPU pico  : {mem['memoria_mb_pico']:.1f} MB")

    # --- Guardado ----------------------------------------------------------
    out_dir = Path(args.out)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    resultado = {
        "modelo": "bert-base-uncased (implementacion propia)",
        "tarea": info.name,
        "config": vars(args),
        "n_train": info.n_train, "n_val": info.n_val, "n_test": info.n_test,
        "max_length": max_length,
        "desempeno_test": desempeno,
        "por_clase": per_class_report(test_true, test_pred, info.class_names),
        "matriz_confusion": confusion(test_true, test_pred),
        "eficiencia": {
            "n_parametros": n_params,
            "tamano_mb": model_size_mb(model),
            **lat, **mem,
        },
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
        "historia": historia,
    }
    destino = out_dir / "metrics" / f"bert_{info.name}.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nResultados guardados en {destino}")


if __name__ == "__main__":
    main()
