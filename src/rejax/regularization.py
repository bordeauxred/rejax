import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

def compute_ortho_loss(params, lambda_coeff, debug=False, log_now=False, exclude_last_layer=True):
    """
    Computes the orthogonalization regularization loss.
    
    Args:
        params: The parameters dictionary (PyTree).
        lambda_coeff: The regularization coefficient.
        debug: Whether to print debug information.
        log_now: Whether to compute expensive diagnostics (singular values).
        exclude_last_layer: Whether to exclude the final layer (assumed to be the output layer).
                            This relies on Flax's default naming (Dense_0, Dense_1, ...).
                            We find the Dense layer with the highest index and exclude it.

    Returns: 
        (total_ortho_loss, layer_metrics)
    """
    # Remove Python control flow on lambda_coeff as it might be traced
    # if lambda_coeff == 0.0:
    #     return 0.0, {}

    flat_params = flatten_dict(params, sep="/")
    ortho_loss = 0.0
    layer_metrics = {}
    
    # Identify layers to regularize
    # We look for keys ending in '/kernel'
    kernel_keys = [k for k in flat_params.keys() if k.endswith('/kernel')]
    
    layers_to_exclude = set()
    if exclude_last_layer:
        # Heuristic: Find the Dense layer with the highest index.
        # Assumes keys look like '.../Dense_(\d+)/kernel'
        max_idx = -1
        last_layer_prefix = None
        
        for k in kernel_keys:
            parts = k.split('/')
            # Look for the part that matches Dense_...
            # Usually the second to last part is the layer name
            if len(parts) >= 2:
                layer_name = parts[-2]
                if layer_name.startswith("Dense_"):
                    try:
                        idx = int(layer_name.split('_')[1])
                        if idx > max_idx:
                            max_idx = idx
                            last_layer_prefix = layer_name
                    except ValueError:
                        pass
        
        if last_layer_prefix:
            layers_to_exclude.add(last_layer_prefix)
            if debug and log_now:
                 jax.debug.print(f"Excluding output layer from ortho loss: {last_layer_prefix}")

    for key_path, value in flat_params.items():
        if not key_path.endswith('/kernel'):
            continue
            
        # Check exclusion
        parts = key_path.split('/')
        layer_name = parts[-2] if len(parts) >= 2 else "unknown"
        
        if layer_name in layers_to_exclude:
            continue

        W = value
        
        # Helper for single matrix
        def _compute_single(w):
            n_in, n_out = w.shape
            if n_in < n_out:
                gram = jnp.matmul(w, w.T)
                identity = jnp.eye(gram.shape[0]) * (n_out / n_in)
            else:
                gram = jnp.matmul(w.T, w)
                identity = jnp.eye(gram.shape[0])
            
            l_loss = jnp.sum((gram - identity) ** 2)
            w_norm = jnp.linalg.norm(w)
            
            def get_singular_stats(g):
                eigvals = jnp.linalg.eigvals(g).real
                singular_values = jnp.sqrt(jnp.maximum(eigvals, 0.0))
                return jnp.max(singular_values), jnp.min(singular_values)

            s_max, s_min = jax.lax.cond(
                log_now,
                get_singular_stats,
                lambda _: (0.0, 0.0),
                operand=gram
            )
            return l_loss, w_norm, s_max, s_min

        # Handle batched parameters (e.g. Ensemble Critics: shape (N, n_in, n_out))
        if W.ndim == 3:
            # vmap over the first dimension (ensemble dimension)
            l_loss, w_norm, s_max, s_min = jax.vmap(_compute_single)(W)
            # Sum loss across ensemble
            layer_ortho_loss = jnp.sum(l_loss)
            # Average metrics for logging
            weight_norm = jnp.mean(w_norm)
            s_max_val = jnp.mean(s_max)
            s_min_val = jnp.mean(s_min)
        elif W.ndim == 2:
            layer_ortho_loss, weight_norm, s_max_val, s_min_val = _compute_single(W)
        else:
            # Skip 1D (biases) or >3D (convs not supported yet?)
            continue

        ortho_loss += layer_ortho_loss
        
        # Use the full path for the metric name, replacing slashes with underscores
        metric_name = "_".join(parts[:-1]) # remove 'kernel'
        layer_metrics[f"diag/s_max_{metric_name}"] = s_max_val
        layer_metrics[f"diag/s_min_{metric_name}"] = s_min_val
        layer_metrics[f"diag/weight_norm_{metric_name}"] = weight_norm
        layer_metrics[f"diag/ortho_loss_{metric_name}"] = layer_ortho_loss

        if debug:
            lambda_l = layer_ortho_loss
            jax.debug.print(f"Layer {key_path}, shape {W.shape}, ortho_loss {{l}}, s_max {{s_max}}, s_min {{s_min}}", 
                            l=lambda_l, s_max=s_max_val, s_min=s_min_val)

    return lambda_coeff * ortho_loss, layer_metrics

