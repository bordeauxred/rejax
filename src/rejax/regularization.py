"""
Orthonormalization utilities for neural networks in JAX.

Provides two modes of regularization:
1. Loss mode: Add Gram deviation as a loss term
2. Optimizer mode: Apply decoupled orthonormalization update after gradient step
"""
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Dict, Optional, Tuple, Any


# =============================================================================
# Core Functions
# =============================================================================

def adaptive_gram_loss(W: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the Gram deviation loss for a weight matrix.

    For wide matrices (n_in < n_out): use W @ W.T with scale = n_out/n_in
    For tall/square matrices: use W.T @ W with scale = 1.0

    Args:
        W: Weight matrix of shape (n_in, n_out)

    Returns:
        (loss, gram_deviation): Frobenius norm loss and the Gram deviation matrix
    """
    n_in, n_out = W.shape

    if n_in < n_out:
        # Wide matrix: enforce row orthogonality
        gram = jnp.matmul(W, W.T)
        scale = n_out / n_in
        target = jnp.eye(n_in) * scale
    else:
        # Tall/square matrix: enforce column orthogonality
        gram = jnp.matmul(W.T, W)
        scale = 1.0
        target = jnp.eye(n_out)

    gram_deviation = gram - target
    loss = jnp.sum(gram_deviation ** 2)

    return loss, gram_deviation


def get_dense_kernels(params: Dict, layer_filter: Optional[callable] = None) -> Dict[str, jnp.ndarray]:
    """
    Extract all Dense layer kernels from a parameter dict.

    Args:
        params: Parameter dictionary (PyTree)
        layer_filter: Optional function(path, kernel) -> bool to filter layers

    Returns:
        Dictionary mapping flattened paths to kernel arrays
    """
    flat_params = flatten_dict(params, sep="/")
    kernels = {}

    for key_path, value in flat_params.items():
        if not key_path.endswith('/kernel'):
            continue
        if value.ndim < 2:
            continue
        if layer_filter is not None and not layer_filter(key_path, value):
            continue
        kernels[key_path] = value

    return kernels


def _find_output_layer_path(kernel_paths: list) -> Optional[str]:
    """
    Find the output layer path from a list of kernel paths.

    Searches for patterns: "output", "final", "head", "action"
    Falls back to the highest Dense_N index.

    Args:
        kernel_paths: List of flattened kernel paths

    Returns:
        The path to the output layer, or None if not found
    """
    output_patterns = ["output", "final", "head", "action"]

    # First, check for explicit naming patterns
    for path in kernel_paths:
        path_lower = path.lower()
        for pattern in output_patterns:
            if pattern in path_lower:
                return path

    # Fallback: find highest Dense_N index
    max_idx = -1
    output_path = None

    for path in kernel_paths:
        parts = path.split('/')
        for part in parts:
            if part.startswith("Dense_"):
                try:
                    idx = int(part.split('_')[1])
                    if idx > max_idx:
                        max_idx = idx
                        output_path = path
                except (ValueError, IndexError):
                    pass

    return output_path


# =============================================================================
# Loss Mode
# =============================================================================

def compute_gram_regularization_loss(
    params: Dict,
    lambda_coeff: float = 0.2,
    exclude_output: bool = True,
    log_diagnostics: bool = False
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Compute the Gram regularization loss for all Dense layers.

    Args:
        params: Parameter dictionary (PyTree)
        lambda_coeff: Regularization coefficient
        exclude_output: Whether to exclude the output layer
        log_diagnostics: Whether to compute expensive diagnostics (singular values)

    Returns:
        (weighted_loss, metrics): The weighted loss and diagnostic metrics
    """
    kernels = get_dense_kernels(params)
    kernel_paths = list(kernels.keys())

    # Identify output layer to exclude
    layers_to_exclude = set()
    if exclude_output and kernel_paths:
        output_path = _find_output_layer_path(kernel_paths)
        if output_path:
            layers_to_exclude.add(output_path)

    total_loss = jnp.array(0.0)
    metrics = {}

    for path, W in kernels.items():
        if path in layers_to_exclude:
            continue

        # Handle batched parameters (ensemble critics: shape (N, n_in, n_out))
        if W.ndim == 3:
            losses, gram_devs = jax.vmap(adaptive_gram_loss)(W)
            layer_loss = jnp.sum(losses)
        elif W.ndim == 2:
            layer_loss, gram_dev = adaptive_gram_loss(W)
        else:
            continue

        total_loss = total_loss + layer_loss

        # Compute metrics
        metric_name = path.replace('/kernel', '').replace('/', '_')
        metrics[f"ortho/loss_{metric_name}"] = layer_loss

        if log_diagnostics:
            if W.ndim == 2:
                # Compute singular value bounds
                n_in, n_out = W.shape
                if n_in < n_out:
                    gram = jnp.matmul(W, W.T)
                else:
                    gram = jnp.matmul(W.T, W)
                eigvals = jnp.linalg.eigvalsh(gram)
                singular_values = jnp.sqrt(jnp.maximum(eigvals, 0.0))
                metrics[f"ortho/s_max_{metric_name}"] = jnp.max(singular_values)
                metrics[f"ortho/s_min_{metric_name}"] = jnp.min(singular_values)
                metrics[f"ortho/weight_norm_{metric_name}"] = jnp.linalg.norm(W)

    metrics["ortho/total_loss"] = total_loss
    weighted_loss = lambda_coeff * total_loss

    return weighted_loss, metrics


# =============================================================================
# Optimizer Mode (Decoupled)
# =============================================================================

def _compute_ortho_gradient(W: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the orthonormalization gradient for a weight matrix.

    For wide matrices (n_in < n_out): grad = 2 * (W @ W.T @ W - scale * W)
    For tall/square matrices: grad = 2 * (W @ W.T @ W - W)

    Args:
        W: Weight matrix of shape (n_in, n_out)

    Returns:
        Orthonormalization gradient
    """
    n_in, n_out = W.shape

    if n_in < n_out:
        # Wide: row orthogonality
        gram = jnp.matmul(W, W.T)  # (n_in, n_in)
        scale = n_out / n_in
        # Gradient: d/dW ||W @ W.T - scale*I||^2 = 2 * (W @ W.T - scale*I) @ W
        #                                        = 2 * (gram @ W - scale * W)
        ortho_grad = 2.0 * (jnp.matmul(gram, W) - scale * W)
    else:
        # Tall/square: column orthogonality
        gram = jnp.matmul(W.T, W)  # (n_out, n_out)
        # Gradient: d/dW ||W.T @ W - I||^2 = 2 * W @ (W.T @ W - I)
        #                                  = 2 * (W @ gram - W)
        ortho_grad = 2.0 * (jnp.matmul(W, gram) - W)

    return ortho_grad


def apply_ortho_update(
    params: Dict,
    lr: float,
    ortho_coeff: float = 1e-3,
    exclude_output: bool = True
) -> Dict:
    """
    Apply decoupled orthonormalization update to parameters.

    This is applied AFTER the regular gradient update.
    Update: W -= lr * ortho_coeff * ortho_grad

    Args:
        params: Parameter dictionary (PyTree)
        lr: Learning rate (same as optimizer)
        ortho_coeff: Orthonormalization coefficient
        exclude_output: Whether to exclude the output layer

    Returns:
        Updated parameters
    """
    flat_params = flatten_dict(params, sep="/")
    kernel_paths = [k for k in flat_params.keys() if k.endswith('/kernel')]

    # Identify output layer to exclude
    layers_to_exclude = set()
    if exclude_output and kernel_paths:
        output_path = _find_output_layer_path(kernel_paths)
        if output_path:
            layers_to_exclude.add(output_path)

    updated_params = dict(flat_params)

    for path in kernel_paths:
        if path in layers_to_exclude:
            continue

        W = flat_params[path]

        if W.ndim == 3:
            # Batched parameters (ensemble)
            ortho_grads = jax.vmap(_compute_ortho_gradient)(W)
            updated_params[path] = W - lr * ortho_coeff * ortho_grads
        elif W.ndim == 2:
            ortho_grad = _compute_ortho_gradient(W)
            updated_params[path] = W - lr * ortho_coeff * ortho_grad

    return unflatten_dict(updated_params, sep="/")


# =============================================================================
# Spectral Diagnostics
# =============================================================================

def compute_network_lipschitz_bound(params: Dict, exclude_output: bool = False) -> jnp.ndarray:
    """
    Compute an upper bound on the network's Lipschitz constant.

    The Lipschitz constant is bounded by the product of spectral norms
    of all weight matrices.

    Args:
        params: Parameter dictionary (PyTree)
        exclude_output: Whether to exclude the output layer

    Returns:
        Upper bound on Lipschitz constant
    """
    kernels = get_dense_kernels(params)
    kernel_paths = list(kernels.keys())

    layers_to_exclude = set()
    if exclude_output and kernel_paths:
        output_path = _find_output_layer_path(kernel_paths)
        if output_path:
            layers_to_exclude.add(output_path)

    lipschitz_bound = jnp.array(1.0)

    for path, W in kernels.items():
        if path in layers_to_exclude:
            continue

        if W.ndim == 2:
            # Compute spectral norm (largest singular value)
            s = jnp.linalg.svd(W, compute_uv=False)
            spectral_norm = jnp.max(s)
            lipschitz_bound = lipschitz_bound * spectral_norm
        elif W.ndim == 3:
            # For ensemble, take max spectral norm across ensemble
            def get_spectral_norm(w):
                s = jnp.linalg.svd(w, compute_uv=False)
                return jnp.max(s)
            spectral_norms = jax.vmap(get_spectral_norm)(W)
            lipschitz_bound = lipschitz_bound * jnp.max(spectral_norms)

    return lipschitz_bound


def compute_spectral_diagnostics(
    params: Dict,
    compute_full_svd: bool = False,
    exclude_output: bool = False
) -> Dict[str, Any]:
    """
    Compute spectral diagnostics for all layers.

    Args:
        params: Parameter dictionary (PyTree)
        compute_full_svd: Whether to compute full SVD (expensive)
        exclude_output: Whether to exclude the output layer

    Returns:
        Dictionary of diagnostic metrics
    """
    kernels = get_dense_kernels(params)
    kernel_paths = list(kernels.keys())

    layers_to_exclude = set()
    if exclude_output and kernel_paths:
        output_path = _find_output_layer_path(kernel_paths)
        if output_path:
            layers_to_exclude.add(output_path)

    metrics = {}
    lipschitz_bound = jnp.array(1.0)

    for path, W in kernels.items():
        if path in layers_to_exclude:
            continue

        metric_name = path.replace('/kernel', '').replace('/', '_')

        if W.ndim == 2:
            if compute_full_svd:
                s = jnp.linalg.svd(W, compute_uv=False)
                metrics[f"spectral/s_max_{metric_name}"] = jnp.max(s)
                metrics[f"spectral/s_min_{metric_name}"] = jnp.min(s)
                metrics[f"spectral/condition_{metric_name}"] = jnp.max(s) / (jnp.min(s) + 1e-8)
                lipschitz_bound = lipschitz_bound * jnp.max(s)
            else:
                # Cheaper: use eigenvalues of Gram matrix
                n_in, n_out = W.shape
                if n_in < n_out:
                    gram = jnp.matmul(W, W.T)
                else:
                    gram = jnp.matmul(W.T, W)
                eigvals = jnp.linalg.eigvalsh(gram)
                singular_values = jnp.sqrt(jnp.maximum(eigvals, 0.0))
                metrics[f"spectral/s_max_{metric_name}"] = jnp.max(singular_values)
                metrics[f"spectral/s_min_{metric_name}"] = jnp.min(singular_values)
                lipschitz_bound = lipschitz_bound * jnp.max(singular_values)

            metrics[f"spectral/weight_norm_{metric_name}"] = jnp.linalg.norm(W)

    metrics["spectral/lipschitz_bound"] = lipschitz_bound

    return metrics


# =============================================================================
# Legacy API (backward compatibility)
# =============================================================================

def compute_ortho_loss(params, lambda_coeff, debug=False, log_now=False, exclude_last_layer=True):
    """
    Legacy API for computing orthogonalization loss.

    This wraps compute_gram_regularization_loss for backward compatibility.
    """
    return compute_gram_regularization_loss(
        params,
        lambda_coeff=lambda_coeff,
        exclude_output=exclude_last_layer,
        log_diagnostics=log_now
    )
