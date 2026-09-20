"""
Verificacion de la implementacion propia de BERT contra HuggingFace.

HuggingFace se usa UNICAMENTE como oraculo: se le pasa la misma entrada a
los dos modelos y se comparan las salidas numero a numero. Si coinciden
dentro de la tolerancia, la implementacion propia es correcta.

Se compara por etapas (embeddings -> capa 1 -> ... -> salida final) para
que, si algo falla, se sepa EXACTAMENTE en que bloque esta el error y no
solo que "el resultado final no cuadra".

Uso (desde la raiz del proyecto):
    python -m bert.verify
"""

import os

# Anaconda y PyTorch traen cada uno su runtime de OpenMP y chocan en Windows.
# Hay que fijarlo ANTES de importar torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from .config import BertConfig
from .model import BertModel
from .classification import BertForSequenceClassification
from .weights import load_pretrained, find_checkpoint, count_parameters, MODEL_ID


def compare(name, mine, theirs, rel_tol=1e-5):
    """
    Compara dos tensores usando error RELATIVO.

    Por que relativo y no absoluto: las activaciones crecen de magnitud ~1
    en los embeddings hasta ~23 en las capas profundas. Una tolerancia
    absoluta fija marcaria como error lo que es simple acumulacion de
    redondeo de float32 (epsilon ~1.2e-7).

    El criterio real de correccion es que el error relativo se mantenga en
    el orden del epsilon de la precision usada. Un bug de implementacion
    haria saltar el error varios ordenes de magnitud de golpe, no creceria
    suavemente.
    """
    diff = (mine - theirs).abs().max().item()
    scale = theirs.abs().max().item()
    rel = diff / max(scale, 1e-12)
    ok = rel < rel_tol
    mark = "OK  " if ok else "FALLA"
    print(f"  [{mark}] {name:28s} dif_abs = {diff:.2e}  dif_rel = {rel:.2e}")
    return ok


def main():
    from transformers import AutoTokenizer, AutoModel

    ckpt = find_checkpoint()
    print(f"Checkpoint: {ckpt}\n")

    # --- Entrada de prueba -------------------------------------------------
    # Dos frases de largo distinto a proposito: asi el batch lleva padding
    # real y se comprueba tambien que la mascara de atencion funciona.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    textos = [
        "the movie was surprisingly good",
        "a dull and pointless waste of time that drags on forever",
    ]
    batch = tokenizer(textos, padding=True, truncation=True,
                      max_length=32, return_tensors="pt")
    print(f"Batch: input_ids {tuple(batch['input_ids'].shape)} (con padding real)\n")

    # --- Los dos modelos ---------------------------------------------------
    config = BertConfig.base()
    mio = BertModel(config)
    load_pretrained(mio, ckpt)
    mio.eval()   # CRITICO: apaga el dropout. Sin esto los numeros no coinciden.

    hf = AutoModel.from_pretrained(MODEL_ID)
    hf.eval()

    print(f"\nParametros mi BERT : {count_parameters(mio):,}")
    print(f"Parametros HF BERT : {sum(p.numel() for p in hf.parameters()):,}")

    # --- Comparacion por etapas -------------------------------------------
    print("\n--- Comparacion capa por capa ---")
    todo_ok = True
    with torch.no_grad():
        seq_mio, pooled_mio, hidden_mio = mio(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            output_hidden_states=True,
        )
        out_hf = hf(**batch, output_hidden_states=True)

    todo_ok &= compare("embeddings", hidden_mio[0], out_hf.hidden_states[0])
    for i in range(1, len(hidden_mio)):
        todo_ok &= compare(f"salida capa {i:2d}", hidden_mio[i], out_hf.hidden_states[i])
    todo_ok &= compare("sequence_output", seq_mio, out_hf.last_hidden_state)
    todo_ok &= compare("pooled_output", pooled_mio, out_hf.pooler_output)

    # --- Prueba del clasificador ------------------------------------------
    print("\n--- Clasificador (cabeza nueva, sin entrenar) ---")
    torch.manual_seed(42)   # la cabeza es aleatoria: fijar semilla para reproducir
    clf = BertForSequenceClassification(config, num_labels=2)
    load_pretrained(clf, ckpt, verbose=False)
    clf.eval()
    with torch.no_grad():
        loss, logits = clf(batch["input_ids"],
                           attention_mask=batch["attention_mask"],
                           labels=torch.tensor([1, 0]))
    print(f"  logits shape : {tuple(logits.shape)}  (esperado (2, 2))")
    print(f"  loss         : {loss.item():.4f}  "
          f"(del orden de ln 2 = 0.69: la cabeza aun no aprendio nada)")
    print(f"  parametros   : {count_parameters(clf):,}")

    print("\n" + "=" * 62)
    if todo_ok:
        print("VERIFICACION SUPERADA: la implementacion propia reproduce")
        print("BERT-base exactamente, con los pesos oficiales cargados.")
    else:
        print("VERIFICACION FALLIDA: revisar las capas marcadas arriba.")
    print("=" * 62)
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
