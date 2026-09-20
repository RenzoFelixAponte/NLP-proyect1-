"""Funciones de activacion usadas por BERT."""

import math

import torch


def gelu(x):
    """
    GELU exacta, definida con la funcion error (erf).

    Es la que usa BERT original. NO confundir con la aproximacion via
    tanh: esa introduce diferencias del orden de 1e-3, suficientes para
    que la verificacion contra la implementacion de referencia falle.

        GELU(x) = x * P(X <= x)  con X ~ N(0, 1)
                = x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
