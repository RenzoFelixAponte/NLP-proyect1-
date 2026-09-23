"""
Comparacion de eficiencia entre modelos, en condiciones controladas.

Por que existe este modulo aparte de las metricas que ya guarda train.py:
cada fine-tuning mide su latencia con la longitud de secuencia de SU dataset
(64 en SST-2, 256 en Yelp), asi que esos numeros NO son comparables entre si.
De hecho salen al reves de lo esperado -- Yelp, con secuencias 4 veces mas
largas, marca menos milisegundos que SST-2 -- porque con batch=1 y secuencias
cortas la GPU esta limitada por el lanzamiento de kernels, no por el calculo.

Aqui se mide cada modelo en la MISMA rejilla de (batch, longitud), en la misma
GPU y en el mismo proceso. Eso es lo que se puede poner en una tabla.

    python -m src.benchmark
    python -m src.benchmark --models bert distilbert --batch-sizes 1 32

La latencia no depende de los pesos, solo de la arquitectura, asi que se mide
sobre los checkpoints pre-entrenados sin necesidad de fine-tuning previo.
"""

import torch                      # antes que numpy: ver nota en train.py

import json
import time
import argparse
from pathlib import Path

from src.paths import RESULTS_DIR
from src.models import build_model, list_models, count_parameters
from src.metrics import measure_latency, measure_gpu_memory, model_size_mb


def benchmark_modelo(nombre_modelo, num_labels, batch_sizes, seq_lens,
                     device, n_warmup=20, n_runs=50):
    """Recorre la rejilla (batch, longitud) para un modelo."""
    model, info = build_model(nombre_modelo, num_labels=num_labels)
    model.to(device).eval()

    resultado = {
        "modelo": nombre_modelo,
        "modelo_nombre": info["nombre"],
        "modelo_hf_id": info["hf_id"],
        "capas": info["capas"],
        "n_parametros": count_parameters(model),
        "tamano_mb": model_size_mb(model),
        "mediciones": [],
    }

    for bs in batch_sizes:
        for sl in seq_lens:
            # Entrada sintetica: la latencia depende de la forma del tensor,
            # no de que tokens concretos lleve dentro. La mascara va toda a 1
            # para medir el caso peor (ningun padding que ignorar).
            input_ids = torch.randint(0, 1000, (bs, sl), device=device)
            attention_mask = torch.ones_like(input_ids)

            lat = measure_latency(model, input_ids, attention_mask, device,
                                  n_warmup=n_warmup, n_runs=n_runs)
            mem = measure_gpu_memory(model, input_ids, attention_mask, device)

            medida = {"batch_size": bs, "seq_len": sl, **lat, **mem}
            # Throughput: muestras por segundo. Es la cifra que importa cuando
            # se procesa en lote; la latencia por si sola solo importa cuando
            # se responde de a una peticion.
            medida["muestras_por_s"] = bs * 1000.0 / lat["latencia_ms_media"]
            resultado["mediciones"].append(medida)

            print(f"  {info['nombre']:16s} batch {bs:>3} x len {sl:>3} | "
                  f"{lat['latencia_ms_media']:7.2f} ms | "
                  f"{medida['muestras_por_s']:8.1f} muestras/s | "
                  f"{mem.get('memoria_mb_pico', 0):7.1f} MB", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list_models())
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--out", default=str(RESULTS_DIR),
                        help="Raiz de salida (por defecto results/ del repo)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombre_gpu = (torch.cuda.get_device_name(0) if device.type == "cuda"
                  else "CPU")

    print("=" * 70)
    print(f"Benchmark de eficiencia | {nombre_gpu}")
    print("=" * 70)

    modelos = [benchmark_modelo(m, args.num_labels, args.batch_sizes,
                                args.seq_lens, device, n_runs=args.n_runs)
               for m in args.models]

    salida = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dispositivo": nombre_gpu,
        "precision": "fp32",
        "config": vars(args),
        "modelos": modelos,
    }

    out_dir = Path(args.out) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / f"benchmark_{time.strftime('%Y%m%d-%H%M%S')}.json"
    destino.write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nGuardado en {destino}")

    # Resumen relativo: es la frase que va al informe ("X veces mas rapido").
    if len(modelos) == 2:
        a, b = modelos
        print(f"\n{b['modelo_nombre']} frente a {a['modelo_nombre']}:")
        print(f"  parametros : {b['n_parametros']/a['n_parametros']:.2f}x")
        for ma, mb in zip(a["mediciones"], b["mediciones"]):
            print(f"  batch {ma['batch_size']:>3} x len {ma['seq_len']:>3} : "
                  f"{ma['latencia_ms_media']/mb['latencia_ms_media']:.2f}x mas rapido, "
                  f"{mb.get('memoria_mb_pico', 0)/max(ma.get('memoria_mb_pico', 1), 1):.2f}x la memoria")


if __name__ == "__main__":
    main()
