"""
Caso de uso: entrenar (destilar) el DistilBERT-custom sobre una tarea de
clasificacion, opcionalmente guiado por un profesor BERT.

Reutiliza el adaptador de datos del proyecto (`src.data_adapter.load_task`)
para no duplicar la logica de carga/normalizacion de SST-2 / AG News / Yelp,
pero el bucle de entrenamiento en si es propio: el modelo que recibe el
gradiente es `DistilBertForSequenceClassification` de `domain/model.py`,
no el de `transformers`.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.custom_distilbert.application.distillation_loss import (
    DistillationWeights,
    distillation_loss,
)
from src.custom_distilbert.domain.model import DistilBertForSequenceClassification
from src.custom_distilbert.infrastructure.teacher import Teacher
from src.metrics import classification_metrics


@dataclass
class DistillTrainConfig:
    epochs: int = 1
    batch_size: int = 16
    lr: float = 5e-5
    max_grad_norm: float = 1.0
    log_every: int = 20
    use_teacher: bool = True
    weights: DistillationWeights = field(default_factory=DistillationWeights)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _build_dataloader(split, tokenizer, max_length, batch_size, shuffle):
    enc = tokenizer(
        list(split["text"]), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    dataset = TensorDataset(
        enc["input_ids"], enc["attention_mask"],
        torch.tensor(list(split["label"]), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class DistillTrainer:
    """Bucle de entrenamiento/destilacion para el estudiante DistilBERT-custom."""

    def __init__(self, student: DistilBertForSequenceClassification,
                 teacher: Optional[Teacher], config: DistillTrainConfig):
        self.student = student.to(config.device)
        self.teacher = teacher
        self.config = config

    def _step_supervised(self, input_ids, attention_mask, labels):
        """Sin profesor: solo cross-entropy contra la etiqueta (fine-tuning normal)."""
        salida = self.student(input_ids=input_ids, attention_mask=attention_mask,
                              labels=labels)
        return {"loss_total": salida.loss, "loss_ce": salida.loss.detach(),
                "loss_kd": torch.tensor(0.0), "loss_cos": torch.tensor(0.0)}

    def _step_distillation(self, input_ids, attention_mask, labels):
        """Con profesor: perdida triple (dura + blanda + coseno)."""
        salida_est = self.student(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=labels, output_hidden_states=True)
        teacher_logits = self.teacher.logits(input_ids, attention_mask)

        # El profesor (BERT, 12 capas) y el estudiante (DistilBERT, 6 capas)
        # no tienen el mismo numero de capas, asi que la perdida coseno solo
        # se aplica sobre el ULTIMO estado oculto de cada uno (misma dim: 768).
        student_last_hidden = salida_est.hidden_states[-1]
        with torch.no_grad():
            teacher_last_hidden = self.teacher.model.base_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

        return distillation_loss(
            student_logits=salida_est.logits, teacher_logits=teacher_logits,
            labels=labels, student_hidden=student_last_hidden,
            teacher_hidden=teacher_last_hidden, attention_mask=attention_mask,
            weights=self.config.weights,
        )

    def _evaluate(self, loader):
        self.student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels in loader:
                input_ids = input_ids.to(self.config.device)
                attention_mask = attention_mask.to(self.config.device)
                salida = self.student(input_ids=input_ids, attention_mask=attention_mask)
                all_preds.append(salida.logits.argmax(dim=-1).cpu())
                all_labels.append(labels)
        preds = torch.cat(all_preds).numpy()
        trues = torch.cat(all_labels).numpy()
        return classification_metrics(trues, preds)

    def fit(self, ds, tokenizer, max_length):
        cfg = self.config
        train_loader = _build_dataloader(ds["train"], tokenizer, max_length,
                                         cfg.batch_size, shuffle=True)
        val_loader = _build_dataloader(ds["validation"], tokenizer, max_length,
                                       cfg.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=cfg.lr)
        step_fn = self._step_distillation if (cfg.use_teacher and self.teacher) \
            else self._step_supervised

        paso = 0
        historia = []
        t0 = time.time()
        for epoca in range(cfg.epochs):
            self.student.train()
            print(f"\n--- Epoca {epoca + 1}/{cfg.epochs} "
                  f"({'destilacion' if step_fn is self._step_distillation else 'supervisado'}) ---")
            for input_ids, attention_mask, labels in train_loader:
                input_ids = input_ids.to(cfg.device)
                attention_mask = attention_mask.to(cfg.device)
                labels = labels.to(cfg.device)

                optimizer.zero_grad(set_to_none=True)
                salida = step_fn(input_ids, attention_mask, labels)
                salida["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(),
                                               cfg.max_grad_norm)
                optimizer.step()

                paso += 1
                if paso % cfg.log_every == 0:
                    registro = {"paso": paso,
                                "loss_total": salida["loss_total"].item(),
                                "loss_ce": salida["loss_ce"].item(),
                                "loss_kd": salida["loss_kd"].item(),
                                "loss_cos": salida["loss_cos"].item()}
                    historia.append(registro)
                    print(f"  paso {paso:5d} | total {registro['loss_total']:.4f} "
                          f"| ce {registro['loss_ce']:.4f} "
                          f"| kd {registro['loss_kd']:.4f} "
                          f"| cos {registro['loss_cos']:.4f}", flush=True)

            metricas_val = self._evaluate(val_loader)
            print(f"  [fin epoca {epoca + 1}] val_acc {metricas_val['accuracy']:.4f} "
                  f"| val_f1 {metricas_val['f1']:.4f}")

        tiempo = time.time() - t0
        print(f"\nEntrenamiento/destilacion completado en {tiempo:.1f} s")
        return {"historia": historia, "tiempo_s": tiempo,
                "val_final": self._evaluate(val_loader)}
