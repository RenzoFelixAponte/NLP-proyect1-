"""
Perdida de destilacion, siguiendo la receta de Hinton et al. (2015) /
Sanh et al. (2019, DistilBERT): una combinacion de TRES terminos.

    L = alpha * L_ce(student, etiqueta_real)          # perdida "dura" supervisada
      + beta  * L_kd(student, teacher, T)             # perdida "blanda" (destilacion)
      + gamma * L_cos(student_hidden, teacher_hidden)  # alinea la geometria interna

1) L_ce: cross-entropy normal contra la etiqueta verdadera. Sin esto el
   estudiante solo aprenderia a imitar al profesor, incluso si el
   profesor se equivoca.

2) L_kd: KL-divergence entre las distribuciones "suavizadas" con
   temperatura T de estudiante y profesor. Softmax con T>1 aplana la
   distribucion y deja ver informacion que el argmax solo no muestra
   (ej. "es 80% positivo pero el profesor duda un poco con 15% negativo"):
   eso es lo que Hinton llama "dark knowledge". Se multiplica por T^2
   porque el gradiente de un softmax con temperatura se achica en ese
   factor, y hay que compensarlo para que el termino no quede diluido.

3) L_cos: coseno entre el ultimo estado oculto del estudiante y el del
   profesor (promediado sobre tokens no-padding). Empuja al estudiante a
   producir representaciones internas alineadas con las del profesor, no
   solo la misma prediccion final. Requiere que ambos tengan la MISMA
   dimension oculta (768 en BERT-base y en DistilBERT-base: coincide).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationWeights:
    alpha_ce: float = 1.0     # peso de la perdida supervisada dura
    beta_kd: float = 1.0      # peso de la perdida de destilacion blanda
    gamma_cos: float = 1.0    # peso de la perdida coseno entre representaciones
    temperature: float = 2.0  # T > 1 suaviza las distribuciones


def kd_soft_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """KL-divergence entre las distribuciones suavizadas del estudiante y el profesor."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kd * (temperature ** 2)


def cosine_hidden_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
    """
    1 - similitud_coseno promedio entre los estados ocultos de cada token
    real (se ignoran los de padding via `attention_mask`).
    """
    mask = attention_mask.unsqueeze(-1).to(student_hidden.dtype)  # (batch, seq, 1)
    student_flat = (student_hidden * mask).reshape(-1, student_hidden.size(-1))
    teacher_flat = (teacher_hidden * mask).reshape(-1, teacher_hidden.size(-1))

    target = torch.ones(student_flat.size(0), device=student_hidden.device)
    return F.cosine_embedding_loss(student_flat, teacher_flat, target)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    weights: DistillationWeights,
) -> dict:
    """Devuelve un diccionario con cada termino por separado y el total,
    para poder loguearlos de forma independiente durante el entrenamiento."""
    ce = F.cross_entropy(student_logits, labels)
    kd = kd_soft_loss(student_logits, teacher_logits, weights.temperature)
    cos = cosine_hidden_loss(student_hidden, teacher_hidden, attention_mask)

    total = weights.alpha_ce * ce + weights.beta_kd * kd + weights.gamma_cos * cos

    return {"loss_total": total, "loss_ce": ce.detach(), "loss_kd": kd.detach(),
            "loss_cos": cos.detach()}
