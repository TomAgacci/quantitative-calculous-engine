# qce_core.py
"""
Core utilities for the Quantitative Calculous Engine (QCE).

Provides:
- Vector type
- FieldSpec type
- Generic finite-difference gradient
- Generic gradient-descent optimizer
"""

from typing import List, Dict, Callable, Tuple

Vector = List[float]
FieldFn = Callable[[Dict, ...], float]
FieldSpec = Tuple[FieldFn, dict]


def finite_difference_grad(
    E_total_fn: Callable[[Vector, List[FieldSpec]], float],
    z: Vector,
    field_specs: List[FieldSpec],
    h: float = 1e-3,
) -> Vector:
    """
    Generic finite-difference gradient of E_total_fn wrt z.
    E_total_fn(z, field_specs) -> scalar.
    """
    base = E_total_fn(z, field_specs)
    grad: Vector = []
    for i in range(len(z)):
        z_pert = z.copy()
        z_pert[i] += h
        val = E_total_fn(z_pert, field_specs)
        grad.append((val - base) / h)
    return grad


def optimize_z(
    E_total_fn: Callable[[Vector, List[FieldSpec]], float],
    z0: Vector,
    field_specs: List[FieldSpec],
    lr: float = 1e-2,
    steps: int = 200,
) -> Vector:
    """
    Generic gradient descent on z to minimize E_total_fn(z, field_specs).
    """
    z = z0.copy()
    for _ in range(steps):
        g = finite_difference_grad(E_total_fn, z, field_specs)
        for i in range(len(z)):
            z[i] -= lr * g[i]
    return z
