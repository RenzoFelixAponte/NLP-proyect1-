"""
Fine-tuning de BERT para clasificacion de texto.

El bucle de entrenamiento esta escrito a mano (no se usa el Trainer de
HuggingFace) para tener control explicito sobre el scheduler, el clipping
y el registro de metricas por iteracion que pide el enunciado.

Los modelos salen de `src/models.py`, con los pesos pre-entrenados.

Uso:
    python -m src.train --task sst2 --epochs 2
    python -m src.train --task sst2 --max-train 15000 --epochs 1

Control termico (util en laptop):
    --max-temp 80 --cooldown 30    pausa 30 s si la GPU pasa de 80 C
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
import subprocess
from pathlib import Path

import numpy as np

from src.paths import RESULTS_DIR
from src.models import (
    build_model,
    build_ablation_model,
    build_tokenizer,
    count_parameters,
)
from src.data_adapter import load_task
from src.metrics import (
    classification_metrics,
    per_class_report,
    confusion,
    measure_latency,
    measure_gpu_memory,
    model_size_mb,
)

SEED = 42


def set_seed(seed=SEED):
    """Fija todas las fuentes de aleatoriedad para que el run sea reproducible."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_temperature():
    """Temperatura de la GPU en grados Celsius, o None si no se puede leer."""
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return int(salida.stdout.strip().split("\n")[0])
    except Exception:
        return None


def control_termico(max_temp, cooldown):
    """
    Pausa el entrenamiento si la GPU se calienta demasiado.

    En una laptop, sostener 85+ C durante media hora castiga tambien a la
    bateria y al VRM. Es preferible alargar el entrenamiento que forzar
    el hardware.
    """
    if not max_temp:
        return
    t = gpu_temperature()
    if t is not None and t >= max_temp:
        print(f"  [termico] GPU a {t} C (limite {max_temp}). "
              f"Pausando {cooldown} s...")
        time.sleep(cooldown)
        t2 = gpu_temperature()
        print(f"  [termico] reanudando a {t2} C")


def build_dataloader(split, tokenizer, max_length, batch_size, shuffle):
    """
    Tokeniza un split completo y lo envuelve en un DataLoader.

    Se tokeniza todo de golpe con padding a max_length fijo: para datasets
    de este tamano cabe en RAM y evita re-tokenizar en cada epoca.
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
            salida = model(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels)

        total_loss += salida.loss.item()
        n_batches += 1
        all_preds.append(salida.logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    return (total_loss / max(n_batches, 1),
            torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert", help="ver src/models.py")
    parser.add_argument("--task", default="sst2", help="sst2 | ag_news | yelp")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="El paper recomienda 5e-5, 3e-5 o 2e-5")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Submuestrear el train (pruebas rapidas o Yelp)")
    parser.add_argument("--max-val", type=int, default=None,
                        help="Submuestrear la validacion. Util en Yelp, donde "
                             "el 10%% del train son 56k ejemplos y evaluar "
                             "cada --eval-every pasos cuesta mas que entrenar")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Por defecto, el recomendado por el dataset")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--no-amp", action="store_true",
                        help="Desactivar precision mixta fp16")
    parser.add_argument("--max-temp", type=int, default=None,
                        help="Pausar si la GPU supera esta temperatura (C)")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Segundos de pausa al superar --max-temp")
    # --- Ablation study (ver README). Si se usa cualquiera de estos, el
    # --- modelo pasa a ser encoder + cabeza propia (src/models.py).
    parser.add_argument("--head-hidden", default=None,
                        help="Capas ocultas de la cabeza, separadas por comas: "
                             "'none' = lineal, '768' = una capa, '512,256' = dos")
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Congelar todo el transformer; solo entrena la cabeza")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Congelar embeddings + las primeras N capas")
    parser.add_argument("--tag", default=None,
                        help="Etiqueta para distinguir corridas "
                             "(ej. 'smoke', 'final', '3epocas')")
    parser.add_argument("--out", default=str(RESULTS_DIR),
                        help="Raiz de salida (por defecto results/ del repo)")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    # --- Datos -------------------------------------------------------------
    subsample = {}
    if args.max_train:
        subsample["train"] = args.max_train
    if args.max_val:
        subsample["validation"] = args.max_val
    subsample = subsample or None
    ds, info_ds = load_task(args.task, subsample=subsample)
    max_length = args.max_length or info_ds.max_length

    tokenizer = build_tokenizer(args.model)

    # Modo ablation: cualquiera de las tres opciones activa la version con
    # cabeza configurable. Sin ellas se usa la cabeza estandar de HuggingFace,
    # que es la que usan las corridas base del informe.
    modo_ablation = (args.head_hidden is not None
                     or args.freeze_encoder
                     or args.freeze_layers > 0)
    if modo_ablation:
        # 'none' (o cadena vacia) = cabeza lineal, sin capa oculta. Se acepta
        # la palabra porque un argumento vacio no siempre sobrevive al paso
        # por sbatch/srun.
        crudo = (args.head_hidden or "").strip().lower()
        head_hidden = ([] if crudo in ("", "none", "0") else
                       [int(h) for h in crudo.split(",") if h.strip()])
        model, info_m, ablacion = build_ablation_model(
            args.model, num_labels=info_ds.num_labels,
            head_hidden=head_hidden, head_dropout=args.head_dropout,
            freeze_encoder=args.freeze_encoder, freeze_layers=args.freeze_layers,
        )
    else:
        model, info_m = build_model(args.model, num_labels=info_ds.num_labels)
        ablacion = None
    model.to(device)

    n_params = count_parameters(model)
    n_params_entrenables = count_parameters(model, only_trainable=True)

    print("=" * 62)
    print(f"Fine-tuning {info_m['nombre']} | tarea: {info_ds.name}")
    print("=" * 62)
    print(f"Dispositivo : {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
    print(f"Precision   : {'fp16 (AMP)' if use_amp else 'fp32'}")
    if args.max_temp:
        print(f"Limite term.: {args.max_temp} C (pausa {args.cooldown} s)")
    print(f"\nDataset     : {info_ds.name} ({info_ds.num_labels} clases)")
    print(f"  train / val / test : {info_ds.n_train:,} / {info_ds.n_val:,} / {info_ds.n_test:,}")
    print(f"  max_length         : {max_length}")
    print(f"\nModelo      : {info_m['hf_id']}")
    print(f"  capas      : {info_m['capas']}")
    print(f"  parametros : {n_params:,} ({model_size_mb(model):.1f} MB en fp32)")
    if ablacion:
        print(f"  cabeza     : {ablacion['head_hidden'] or 'lineal'} "
              f"(dropout {ablacion['head_dropout']})")
        print(f"  congelado  : encoder={ablacion['freeze_encoder']} "
              f"capas={ablacion['freeze_layers']}")
        print(f"  entrenables: {n_params_entrenables:,} "
              f"({100 * n_params_entrenables / n_params:.1f}% del total)")

    train_loader = build_dataloader(ds["train"], tokenizer, max_length,
                                    args.batch_size, shuffle=True)
    val_loader = build_dataloader(ds["validation"], tokenizer, max_length,
                                  args.batch_size, shuffle=False)
    test_loader = build_dataloader(ds["test"], tokenizer, max_length,
                                   args.batch_size, shuffle=False)

    # --- Optimizador -------------------------------------------------------
    # No se aplica weight decay a bias ni a LayerNorm: son terminos de
    # escala/desplazamiento, penalizarlos degrada el modelo.
    # Solo los parametros entrenables: si se congelo parte del encoder, sus
    # pesos no deben llegar al optimizador.
    no_decay = ["bias", "LayerNorm.weight"]
    entrenables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped = [
        {"params": [p for n, p in entrenables
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in entrenables
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
    # Metricas de validacion al cierre de cada epoca. Son las que deciden que
    # configuracion gana el ablation study: el test se mira una sola vez, al
    # final, y elegir con el seria elegir sobre el conjunto que se reporta.
    historia_epocas = []
    paso_global = 0
    mejor_f1 = -1.0
    mejor_epoca = None
    mejor_estado = None
    t0 = time.time()

    for epoca in range(args.epochs):
        print(f"\n--- Epoca {epoca + 1}/{args.epochs} ---", flush=True)
        model.train()
        acum_loss, acum_n = 0.0, 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                salida = model(input_ids=input_ids, attention_mask=attention_mask,
                               labels=labels)
            loss = salida.loss

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
                          f"| val_loss {val_loss:.4f} | val_acc {val_acc:.4f}",
                          flush=True)
                    model.train()
                else:
                    print(f"  paso {paso_global:5d} | train_loss {train_loss:.4f}",
                          flush=True)

                historia.append(registro)
                control_termico(args.max_temp, args.cooldown)

        # Evaluacion al cierre de cada epoca
        val_loss, val_pred, val_true = evaluate(model, val_loader, device, use_amp)
        m = classification_metrics(val_true, val_pred)
        print(f"  [fin epoca {epoca+1}] val_loss {val_loss:.4f} | "
              f"acc {m['accuracy']:.4f} | f1 {m['f1']:.4f}", flush=True)

        historia_epocas.append({"epoca": epoca + 1, "val_loss": val_loss, **m})

        if m["f1"] > mejor_f1:
            mejor_f1 = m["f1"]
            mejor_epoca = epoca + 1
            mejor_estado = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            print(f"  -> mejor modelo hasta ahora (f1 = {mejor_f1:.4f})", flush=True)

    tiempo_entrenamiento = time.time() - t0
    print(f"\nEntrenamiento completado en {tiempo_entrenamiento/60:.1f} min")

    # --- Evaluacion final en TEST con el mejor checkpoint -----------------
    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)

    test_loss, test_pred, test_true = evaluate(model, test_loader, device, use_amp)
    desempeno = classification_metrics(test_true, test_pred)

    print("\n" + "=" * 62)
    print(f"RESULTADOS EN TEST | {info_m['nombre']} | {info_ds.name}")
    print("=" * 62)
    print(f"  Accuracy  : {desempeno['accuracy']:.4f}")
    print(f"  Precision : {desempeno['precision']:.4f}")
    print(f"  Recall    : {desempeno['recall']:.4f}")
    print(f"  F1-score  : {desempeno['f1']:.4f}")

    print("\n  Por clase:")
    for fila in per_class_report(test_true, test_pred, info_ds.class_names):
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "modelo": args.model,
        "modelo_nombre": info_m["nombre"],
        "modelo_hf_id": info_m["hf_id"],
        "capas": info_m["capas"],
        "tarea": info_ds.name,
        "config": vars(args),
        "n_train": info_ds.n_train, "n_val": info_ds.n_val, "n_test": info_ds.n_test,
        "max_length": max_length,
        "desempeno_test": desempeno,
        "desempeno_val": {"mejor_f1": mejor_f1, "mejor_epoca": mejor_epoca,
                          "por_epoca": historia_epocas},
        "por_clase": per_class_report(test_true, test_pred, info_ds.class_names),
        "matriz_confusion": confusion(test_true, test_pred),
        "ablacion": ablacion,
        "eficiencia": {
            "n_parametros": n_params,
            "n_parametros_entrenables": n_params_entrenables,
            "tamano_mb": model_size_mb(model),
            **lat, **mem,
        },
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
        "historia": historia,
    }
    # Nombre UNICO por corrida: modelo_tarea[_tag]_fecha-hora.json
    # Nunca se sobrescribe nada. Antes los archivos se llamaban solo
    # modelo_tarea.json y una prueba rapida borraba el resultado bueno.
    partes = [args.model, info_ds.name]
    if args.tag:
        partes.append(args.tag)
    partes.append(time.strftime("%Y%m%d-%H%M%S"))
    destino = out_dir / "metrics" / ("_".join(partes) + ".json")

    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nResultados guardados en {destino}")


if __name__ == "__main__":
    main()
