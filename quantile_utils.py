from typing import Any, Literal, NamedTuple, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

NDFloatArray = NDArray[np.floating[Any]]
NDIntArray = NDArray[np.integer[Any]]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class GenerateOutput(NamedTuple):
    output_ids_BT: NDIntArray
    logits_BV: torch.Tensor
    tokenwise_log_probs: list[tuple[NDIntArray, NDFloatArray]]
    continuations: list[str]


class TopKResult(NamedTuple):
    indices: list[int]
    probs: list[float]

QUANTILE_KEYS = (
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1 - 1e-4,
    1 - 1e-5,
    1 - 1e-6,
    1 - 1e-7,
    1 - 1e-8,
)

def approximate_quantile(
    q: float,
    N: int,
    k: int,
    bottom_k_values: NDFloatArray,
    top_k_values: NDFloatArray,
) -> NDFloatArray:
    """
    Approximate the q-quantile for each batch, given the bottom k and top k values.

    Parameters:
    - q: The desired quantile (cumulative probability).
    - N: The total number of data points.
    - k: The number of known bottom and top values.
    - bottom_k_values: Array of shape (batch_size, k) containing bottom k values.
    - top_k_values: Array of shape (batch_size, k) containing top k values.

    Returns:
    - approx_values: Array of shape (batch_size,) with the approximated quantile values.
    """
    batch_size = bottom_k_values.shape[0]
    approx_values = np.empty(batch_size, dtype=np.float64)

    # Known cumulative probabilities for bottom_k_values and top_k_values
    bottom_p = np.arange(1, k + 1) / N  # Shape: (k,)
    top_p = (N - k + np.arange(1, k + 1)) / N  # Shape: (k,)

    # Determine if q is in lower or upper quantile range
    if (1 / N) <= q <= (k / N):
        # Lower quantiles
        p = bottom_p
        values = bottom_k_values
    elif ((N - k + 1) / N) <= q <= 1:
        # Upper quantiles
        p = top_p
        values = top_k_values
    else:
        raise ValueError(f"q={q} is out of the known quantile ranges based on k={k} and N={N}.")

    # Find the indices for interpolation
    indices = np.searchsorted(p, q, side="right") - 1
    indices = np.clip(indices, 0, k - 2)  # Ensure indices are within valid range

    # Get the cumulative probabilities and values for interpolation
    p_lower = p[indices]  # Shape: (batch_size,)
    p_upper = p[indices + 1]  # Shape: (batch_size,)
    v_lower = values[:, indices]  # Shape: (batch_size,)
    v_upper = values[:, indices + 1]  # Shape: (batch_size,)

    # Compute the fraction for interpolation
    fraction = (v_upper - v_lower) / (p_upper - p_lower)

    # Handle cases where p_upper == p_lower to avoid division by zero
    zero_denominator = p_upper == p_lower
    approx_values[zero_denominator] = v_lower[zero_denominator]
    approx_values[~zero_denominator] = v_lower[~zero_denominator] + fraction * (
        q - p_lower[~zero_denominator]
    )

    return approx_values