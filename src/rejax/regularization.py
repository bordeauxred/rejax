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

    # Handle edge case of zero dimensions
    if n_in == 0 or n_out == 0:
        return jnp.array(0.0), jnp.zeros((1, 1))

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


# =============================================================================
# L2-Init Regularization (Regenerative Regularization)
# =============================================================================

def compute_l2_init_loss(
    params: Dict,
    init_params: Dict,
    lambda_coeff: float = 0.001,
    exclude_output: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Compute L2 regularization loss towards initial parameters.

    This implements "regenerative regularization" from Lyle et al. (2023):
    loss = λ * ||θ - θ₀||²

    Unlike standard L2 (towards zero), this preserves the initial weight
    distribution and prevents rank collapse.

    Args:
        params: Current parameter dictionary (PyTree)
        init_params: Initial parameter dictionary (PyTree, frozen at init)
        lambda_coeff: Regularization coefficient (recommended: 0.001)
        exclude_output: Whether to exclude the output layer

    Returns:
        (weighted_loss, metrics): The weighted loss and diagnostic metrics
    """
    kernels = get_dense_kernels(params)
    init_kernels = get_dense_kernels(init_params)
    kernel_paths = list(kernels.keys())

    # Identify output layer to exclude
    layers_to_exclude = set()
    if exclude_output and kernel_paths:
        output_path = _find_output_layer_path(kernel_paths)
        if output_path:
            layers_to_exclude.add(output_path)

    total_distance = jnp.array(0.0)
    metrics = {}

    for path in kernel_paths:
        if path in layers_to_exclude:
            continue
        if path not in init_kernels:
            continue

        W = kernels[path]
        W_init = init_kernels[path]

        # Handle batched parameters (ensemble critics: shape (N, n_in, n_out))
        if W.ndim == 3:
            # Sum over ensemble, compute squared L2 distance
            distances = jax.vmap(lambda w, w0: jnp.sum((w - w0) ** 2))(W, W_init)
            layer_distance = jnp.sum(distances)
        elif W.ndim == 2:
            layer_distance = jnp.sum((W - W_init) ** 2)
        else:
            continue

        total_distance = total_distance + layer_distance

        # Per-layer metrics
        metric_name = path.replace('/kernel', '').replace('/', '_')
        metrics[f"l2_init/distance_{metric_name}"] = layer_distance

    metrics["l2_init/total_distance"] = total_distance
    weighted_loss = lambda_coeff * total_distance

    return weighted_loss, metrics


# =============================================================================
# Scale-AdaMO: Per-layer learnable scaling
# =============================================================================

def get_scale_params(params: Dict) -> Dict[str, jnp.ndarray]:
    """
    Extract all scale parameters from a parameter dict.

    Scale parameters are named 'scale_*' (e.g., 'scale_0', 'scale_1', 'scale_conv').

    Args:
        params: Parameter dictionary (PyTree)

    Returns:
        Dictionary mapping flattened paths to scale parameter arrays
    """
    flat_params = flatten_dict(params, sep="/")
    scales = {}

    for key_path, value in flat_params.items():
        # Match scale parameters (scale_0, scale_1, scale_conv, etc.)
        parts = key_path.split('/')
        if any(part.startswith('scale_') or part == 'scale' for part in parts):
            scales[key_path] = value

    return scales


def compute_scale_regularization_loss(
    params: Dict,
    reg_coeff: float = 0.01,
    exclude_output: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Compute log(α)² regularization to encourage α ≈ 1.

    This regularization prevents scale parameters from collapsing to 0
    or exploding to large values, while allowing the network to learn
    appropriate scaling for different reward magnitudes.

    Args:
        params: Parameter dictionary (PyTree)
        reg_coeff: Regularization coefficient (default: 0.01)
        exclude_output: Whether to exclude output layer scales (unused for scale params)

    Returns:
        (weighted_loss, metrics): The weighted loss and diagnostic metrics
    """
    scales = get_scale_params(params)

    total_loss = jnp.array(0.0)
    metrics = {}
    scale_values = []

    for path, alpha in scales.items():
        # log(α)² encourages α near 1, penalizes extreme values
        # Use absolute value to handle negative scales (shouldn't happen but safety)
        layer_loss = jnp.sum(jnp.log(jnp.abs(alpha) + 1e-8) ** 2)
        total_loss = total_loss + layer_loss

        # Per-layer metrics
        metric_name = path.replace('/', '_')
        metrics[f"scale/{metric_name}"] = jnp.mean(alpha)
        scale_values.append(jnp.mean(alpha))

    # Aggregate metrics
    if scale_values:
        scale_array = jnp.array(scale_values)
        metrics["scale/mean"] = jnp.mean(scale_array)
        metrics["scale/min"] = jnp.min(scale_array)
        metrics["scale/max"] = jnp.max(scale_array)
        # Total Lipschitz contribution from scales: product of all α
        metrics["scale/lipschitz"] = jnp.prod(scale_array)

    metrics["scale/reg_loss"] = total_loss
    weighted_loss = reg_coeff * total_loss

    return weighted_loss, metrics


# =============================================================================
# NaP (Normalize-and-Project)
# =============================================================================

def compute_weight_norms(
    params: Dict,
    exclude_output: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Compute Frobenius norms of kernel weights.

    Args:
        params: Parameter dictionary (PyTree)
        exclude_output: Whether to exclude the output layer

    Returns:
        Dictionary mapping kernel paths to their Frobenius norms
    """
    kernels = get_dense_kernels(params)
    kernel_paths = list(kernels.keys())

    # Identify output layer to exclude
    layers_to_exclude = set()
    if exclude_output and kernel_paths:
        output_path = _find_output_layer_path(kernel_paths)
        if output_path:
            layers_to_exclude.add(output_path)

    norms = {}
    for path, W in kernels.items():
        if path in layers_to_exclude:
            continue

        if W.ndim == 3:
            # Ensemble: compute norm for each member
            norms[path] = jax.vmap(jnp.linalg.norm)(W)
        elif W.ndim == 2:
            norms[path] = jnp.linalg.norm(W)

    return norms


def apply_nap_projection(
    params: Dict,
    init_norms: Dict[str, jnp.ndarray],
    exclude_output: bool = True,
) -> Dict:
    """
    Apply NaP (Normalize-and-Project) to maintain initial weight norms.

    After each optimizer step, projects weights to maintain their initial norms:
    W ← (ρₗ * W) / ||W||

    where ρₗ is the initial norm of layer l.

    This decouples the effective learning rate from parameter norm growth,
    which helps maintain plasticity in continual learning.

    Args:
        params: Parameter dictionary (PyTree)
        init_norms: Dictionary of initial norms from compute_weight_norms()
        exclude_output: Whether to exclude the output layer

    Returns:
        Updated parameters with projected weights
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
        if path not in init_norms:
            continue

        W = flat_params[path]
        target_norm = init_norms[path]

        if W.ndim == 3:
            # Ensemble: project each member separately
            def project_single(w, rho):
                current_norm = jnp.linalg.norm(w)
                # Avoid division by zero
                scale = rho / (current_norm + 1e-8)
                return w * scale
            updated_params[path] = jax.vmap(project_single)(W, target_norm)
        elif W.ndim == 2:
            current_norm = jnp.linalg.norm(W)
            scale = target_norm / (current_norm + 1e-8)
            updated_params[path] = W * scale

    return unflatten_dict(updated_params, sep="/")


def compute_wsn_loss(
    params: Dict,
    lambda_coeff: float = 0.001,
    num_power_iters: int = 1,
    exclude_output: bool = True,
    rng_key: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Compute Weight Spectral Norm (WSN) regularization loss.

    Drives the squared spectral norm of each weight matrix towards 1:
    loss = λ * Σ (σ²(W) - 1)²

    where σ(W) is the largest singular value (spectral norm) of W,
    estimated via power iteration.

    Reference: Lewandowski et al. 2025

    Args:
        params: Parameter dictionary (PyTree)
        lambda_coeff: Regularization coefficient (recommended: 0.001)
        num_power_iters: Number of power iteration steps (1 is usually enough)
        exclude_output: Whether to exclude the output layer
        rng_key: Optional JAX random key for initializing power iteration

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

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    total_penalty = jnp.array(0.0)
    metrics = {}

    for path in kernel_paths:
        if path in layers_to_exclude:
            continue

        W = kernels[path]

        # Handle ensemble dimension (vmapped parameters)
        if W.ndim == 3:
            # For ensembles, average the penalty across members
            def single_wsn_penalty(w, key):
                return _compute_single_wsn_penalty(w, num_power_iters, key)
            rng_key, *subkeys = jax.random.split(rng_key, W.shape[0] + 1)
            penalties = jax.vmap(single_wsn_penalty)(W, jnp.array(subkeys))
            penalty = jnp.mean(penalties)
        elif W.ndim == 2:
            rng_key, subkey = jax.random.split(rng_key)
            penalty, _ = _compute_single_wsn_penalty_with_sigma(W, num_power_iters, subkey)
        else:
            continue

        total_penalty = total_penalty + penalty

        # Store per-layer metrics
        layer_name = path.replace('/kernel', '').replace('params/', '')
        metrics[f'wsn/{layer_name}/penalty'] = penalty

    weighted_loss = lambda_coeff * total_penalty
    metrics['wsn/total_penalty'] = total_penalty
    metrics['wsn/weighted_loss'] = weighted_loss

    return weighted_loss, metrics


def _compute_single_wsn_penalty(
    W: jnp.ndarray,
    num_power_iters: int,
    rng_key: jnp.ndarray,
) -> jnp.ndarray:
    """Compute WSN penalty for a single weight matrix."""
    penalty, _ = _compute_single_wsn_penalty_with_sigma(W, num_power_iters, rng_key)
    return penalty


def _compute_single_wsn_penalty_with_sigma(
    W: jnp.ndarray,
    num_power_iters: int,
    rng_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute WSN penalty and estimated sigma² for a single weight matrix.

    Uses power iteration on W^T @ W to estimate the largest singular value squared.
    """
    n_in, n_out = W.shape

    # Initialize random vector for power iteration
    v = jax.random.normal(rng_key, (n_out,))
    v = v / (jnp.sqrt(jnp.sum(v ** 2)) + 1e-12)

    def power_iter_step(v, _):
        # v_new = W^T @ (W @ v)
        u = W @ v
        v_new = W.T @ u
        v_norm = jnp.sqrt(jnp.sum(v_new ** 2) + 1e-12)
        return v_new / v_norm, None

    v, _ = jax.lax.scan(power_iter_step, v, None, length=num_power_iters)

    # Estimate sigma² = ||W @ v||² (Rayleigh quotient since v is unit norm)
    u_final = W @ v
    sigma_sq = jnp.sum(u_final ** 2)

    # Penalty: (σ² - 1)²
    penalty = (sigma_sq - 1.0) ** 2

    return penalty, sigma_sq
