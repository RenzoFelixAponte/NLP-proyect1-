"""
Envoltorio del modelo "profesor" (teacher) para la destilacion.

El profesor SI se deja como el `BertForSequenceClassification` normal de
`transformers` -- no hace falta reimplementarlo, porque nunca se le hace
backward (`torch.no_grad()` siempre): solo se usa para generar los
logits "blandos" (soft labels) que el estudiante intenta imitar. El unico
modelo que de verdad se entrena con gradiente es el estudiante custom.

Idealmente el profesor ya deberia estar afinado (fine-tuned) en la tarea
-- si no, sus logits no dicen nada util sobre la tarea. Este wrapper
soporta cargar un `--teacher-checkpoint` local (un .pt guardado por este
mismo proyecto); si no se pasa ninguno, se usa BERT pre-entrenado con una
cabeza nueva, valido solo para probar que el pipeline de destilacion
corre de punta a punta (smoke test), no para resultados reales.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class Teacher:
    """Envuelve un BertForSequenceClassification en modo solo-inferencia."""

    def __init__(self, num_labels: int, hf_id: str = "bert-base-uncased",
                 checkpoint_path: Optional[str] = None, device: str = "cpu"):
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_id, num_labels=num_labels
        )
        if checkpoint_path:
            ruta = Path(checkpoint_path)
            if not ruta.exists():
                raise FileNotFoundError(f"No existe el checkpoint del profesor: {ruta}")
            estado = torch.load(ruta, map_location="cpu")
            self.model.load_state_dict(estado)
            print(f"[teacher] pesos afinados cargados desde {ruta}")
        else:
            print("[teacher] usando BERT pre-entrenado SIN afinar en la tarea "
                  "(valido solo para smoke test, no para destilacion real)")

        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        salida = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return salida.logits
