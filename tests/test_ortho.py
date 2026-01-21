"""
Tests for orthonormalization utilities and PPO integration.
"""
import jax
import jax.numpy as jnp
import numpy as np

from rejax.regularization import (
    adaptive_gram_loss,
    get_dense_kernels,
    _find_output_layer_path,
    compute_gram_regularization_loss,
    apply_ortho_update,
    compute_ortho_loss,  # Legacy API
)


# =============================================================================
# Core Function Tests
# =============================================================================

def test_adaptive_gram_loss_orthogonal():
    """Orthogonal matrix should have zero loss."""
    key = jax.random.PRNGKey(0)
    W = jax.random.orthogonal(key, 4)  # 4x4 orthogonal

    loss, gram_dev = adaptive_gram_loss(W)

    assert jnp.allclose(loss, 0.0, atol=1e-5)
    assert jnp.allclose(gram_dev, 0.0, atol=1e-5)


def test_adaptive_gram_loss_wide():
    """Wide matrix: W @ W.T = scale * I."""
    # W = [[1, 0, 1, 0], [0, 1, 0, 1]] -> 2x4
    # W @ W.T = [[2, 0], [0, 2]]
    # n_in=2, n_out=4 -> scale = 4/2 = 2
    # target = 2 * I = [[2, 0], [0, 2]]
    # So loss should be 0

    W = jnp.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    loss, gram_dev = adaptive_gram_loss(W)

    assert jnp.allclose(loss, 0.0, atol=1e-5)


def test_adaptive_gram_loss_tall():
    """Tall matrix: W.T @ W = I."""
    # Create a tall orthonormal matrix
    key = jax.random.PRNGKey(1)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (4, 2)))

    loss, gram_dev = adaptive_gram_loss(Q)

    assert jnp.allclose(loss, 0.0, atol=1e-5)


def test_adaptive_gram_loss_non_orthogonal():
    """Non-orthogonal matrix should have non-zero loss."""
    W = jnp.ones((2, 2))  # Definitely not orthogonal

    loss, gram_dev = adaptive_gram_loss(W)

    assert loss > 0


def test_get_dense_kernels():
    """Test extraction of Dense kernels from params."""
    params = {
        "Dense_0": {"kernel": jnp.ones((4, 8)), "bias": jnp.zeros(8)},
        "Dense_1": {"kernel": jnp.ones((8, 2)), "bias": jnp.zeros(2)},
    }

    kernels = get_dense_kernels(params)

    assert len(kernels) == 2
    assert "Dense_0/kernel" in kernels
    assert "Dense_1/kernel" in kernels


def test_get_dense_kernels_with_filter():
    """Test filtering of Dense kernels."""
    params = {
        "Dense_0": {"kernel": jnp.ones((4, 8)), "bias": jnp.zeros(8)},
        "Dense_1": {"kernel": jnp.ones((8, 2)), "bias": jnp.zeros(2)},
    }

    # Only get first layer
    kernels = get_dense_kernels(
        params,
        layer_filter=lambda path, _: "Dense_0" in path
    )

    assert len(kernels) == 1
    assert "Dense_0/kernel" in kernels


def test_find_output_layer_path_by_pattern():
    """Test finding output layer by name pattern."""
    paths = ["params/Dense_0/kernel", "params/action_head/kernel"]

    result = _find_output_layer_path(paths)

    # Should find "action" pattern
    assert result == "params/action_head/kernel"


def test_find_output_layer_path_by_index():
    """Test finding output layer by highest Dense index."""
    paths = ["params/Dense_0/kernel", "params/Dense_1/kernel", "params/Dense_2/kernel"]

    result = _find_output_layer_path(paths)

    assert result == "params/Dense_2/kernel"


# =============================================================================
# Loss Mode Tests
# =============================================================================

def test_compute_gram_regularization_loss():
    """Test the main regularization loss function."""
    W0 = jax.random.orthogonal(jax.random.PRNGKey(0), 4)
    W1 = jnp.ones((4, 2))  # Non-orthogonal

    params = {
        "Dense_0": {"kernel": W0},
        "Dense_1": {"kernel": W1},
    }

    loss, metrics = compute_gram_regularization_loss(
        params,
        lambda_coeff=1.0,
        exclude_output=False
    )

    # Loss should be non-zero due to W1
    assert loss > 0
    assert "ortho/total_loss" in metrics


def test_compute_gram_regularization_loss_exclude_output():
    """Test exclusion of output layer."""
    W0 = jnp.ones((2, 2))  # Non-orthogonal
    W1 = jnp.ones((2, 2))  # Non-orthogonal (output)

    params = {
        "Dense_0": {"kernel": W0},
        "Dense_1": {"kernel": W1},
    }

    loss_excl, _ = compute_gram_regularization_loss(
        params, lambda_coeff=1.0, exclude_output=True
    )
    loss_all, _ = compute_gram_regularization_loss(
        params, lambda_coeff=1.0, exclude_output=False
    )

    # Excluding output should give lower loss
    assert loss_excl < loss_all


def test_legacy_api_compute_ortho_loss():
    """Test backward-compatible legacy API."""
    W0 = jnp.ones((2, 2))
    params = {"Dense_0": {"kernel": W0}}

    loss, metrics = compute_ortho_loss(
        params, lambda_coeff=1.0, exclude_last_layer=False
    )

    assert loss > 0


# =============================================================================
# Optimizer Mode Tests
# =============================================================================

def test_apply_ortho_update_reduces_loss():
    """Ortho update should reduce the Gram deviation loss."""
    key = jax.random.PRNGKey(42)
    W = jax.random.normal(key, (4, 4))  # Random, not orthogonal

    params = {"Dense_0": {"kernel": W}}

    # Compute initial loss
    loss_before, _ = compute_gram_regularization_loss(
        params, lambda_coeff=1.0, exclude_output=False
    )

    # Apply ortho update
    updated_params = apply_ortho_update(
        params, lr=0.1, ortho_coeff=0.1, exclude_output=False
    )

    # Compute loss after update
    loss_after, _ = compute_gram_regularization_loss(
        updated_params, lambda_coeff=1.0, exclude_output=False
    )

    # Loss should decrease
    assert loss_after < loss_before


def test_apply_ortho_update_preserves_orthogonal():
    """Ortho update should preserve already orthogonal matrices."""
    key = jax.random.PRNGKey(0)
    W = jax.random.orthogonal(key, 4)

    params = {"Dense_0": {"kernel": W}}

    updated_params = apply_ortho_update(
        params, lr=0.1, ortho_coeff=0.1, exclude_output=False
    )

    # Should still be close to orthogonal
    W_updated = updated_params["Dense_0"]["kernel"]
    gram = W_updated.T @ W_updated
    identity = jnp.eye(4)

    assert jnp.allclose(gram, identity, atol=1e-3)


def test_apply_ortho_update_excludes_output():
    """Ortho update should exclude output layer when specified."""
    W0 = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
    W1 = jax.random.normal(jax.random.PRNGKey(1), (4, 2))

    params = {
        "Dense_0": {"kernel": W0},
        "Dense_1": {"kernel": W1},  # Output layer
    }

    updated_params = apply_ortho_update(
        params, lr=0.1, ortho_coeff=0.1, exclude_output=True
    )

    # Dense_1 should be unchanged
    assert jnp.allclose(
        updated_params["Dense_1"]["kernel"],
        params["Dense_1"]["kernel"]
    )

    # Dense_0 should be updated
    assert not jnp.allclose(
        updated_params["Dense_0"]["kernel"],
        params["Dense_0"]["kernel"]
    )


def test_apply_ortho_update_batched():
    """Test ortho update with batched (ensemble) parameters."""
    key = jax.random.PRNGKey(0)
    W_batched = jax.random.normal(key, (2, 4, 4))  # 2 ensemble members

    params = {"Dense_0": {"kernel": W_batched}}

    updated_params = apply_ortho_update(
        params, lr=0.1, ortho_coeff=0.1, exclude_output=False
    )

    # Both ensemble members should be updated
    assert updated_params["Dense_0"]["kernel"].shape == (2, 4, 4)
    assert not jnp.allclose(
        updated_params["Dense_0"]["kernel"],
        params["Dense_0"]["kernel"]
    )


# =============================================================================
# PPO Integration Tests
# =============================================================================

def test_ppo_with_ortho_loss_mode():
    """Test PPO creation with ortho loss mode."""
    from rejax import PPO

    ppo = PPO.create(
        env="CartPole-v1",
        ortho_mode="loss",
        ortho_lambda=0.2,
        total_timesteps=1000,
    )

    assert ppo.ortho_mode == "loss"
    assert ppo.ortho_lambda == 0.2


def test_ppo_with_ortho_optimizer_mode():
    """Test PPO creation with ortho optimizer mode."""
    from rejax import PPO

    ppo = PPO.create(
        env="CartPole-v1",
        ortho_mode="optimizer",
        ortho_coeff=1e-3,
        total_timesteps=1000,
    )

    assert ppo.ortho_mode == "optimizer"
    assert ppo.ortho_coeff == 1e-3


def test_ppo_with_groupsort_activation():
    """Test PPO creation with groupsort activation."""
    from rejax import PPO

    ppo = PPO.create(
        env="CartPole-v1",
        agent_kwargs={"activation": "groupsort", "hidden_layer_sizes": (64, 64)},
        total_timesteps=1000,
    )

    # Should create successfully
    assert ppo is not None


def test_ppo_train_with_ortho_loss():
    """Test that PPO trains successfully with ortho loss mode."""
    from rejax import PPO

    ppo = PPO.create(
        env="CartPole-v1",
        ortho_mode="loss",
        ortho_lambda=0.2,
        total_timesteps=1000,
        num_envs=4,
    )

    key = jax.random.PRNGKey(0)
    ts, _ = PPO.train(ppo, key)

    # Training should complete
    assert ts.global_step > 0


def test_ppo_train_with_ortho_optimizer():
    """Test that PPO trains successfully with ortho optimizer mode."""
    from rejax import PPO

    ppo = PPO.create(
        env="CartPole-v1",
        ortho_mode="optimizer",
        ortho_coeff=1e-3,
        total_timesteps=1000,
        num_envs=4,
    )

    key = jax.random.PRNGKey(0)
    ts, _ = PPO.train(ppo, key)

    # Training should complete
    assert ts.global_step > 0


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

def test_ortho_loss_zero_for_orthogonal():
    """Legacy test: Orthogonal matrix should have zero loss."""
    key = jax.random.PRNGKey(0)
    W = jax.random.orthogonal(key, 4)

    params = {"Dense_0": {"kernel": W}}

    loss, metrics = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=False)

    assert jnp.allclose(loss, 0.0, atol=1e-5)


def test_ortho_loss_wide():
    """Legacy test: Wide matrix with proper scaling."""
    W = jnp.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    params = {"Dense_0": {"kernel": W}}

    loss, metrics = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=False)
    assert jnp.allclose(loss, 0.0, atol=1e-5)


def test_exclude_last_layer():
    """Legacy test: Exclude last layer from regularization."""
    W0 = jnp.ones((2, 2))
    W1 = jnp.ones((2, 2))

    params = {
        "Dense_0": {"kernel": W0},
        "Dense_1": {"kernel": W1}
    }

    loss_excl, _ = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=True)
    loss_all, _ = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=False)

    # W0: W0.T @ W0 = [[2,2],[2,2]]. I = [[1,0],[0,1]].
    # Diff = [[1,2],[2,1]]. Sq = [[1,4],[4,1]]. Sum = 10.
    assert jnp.allclose(loss_excl, 10.0, atol=1e-5)
    assert jnp.allclose(loss_all, 20.0, atol=1e-5)


def test_ortho_loss_batched():
    """Legacy test: Batched (ensemble) parameters."""
    W0 = jax.random.orthogonal(jax.random.PRNGKey(0), 4)
    W1 = jnp.eye(4) * np.sqrt(2)  # W.T @ W = 2I, so loss = ||2I - I||^2 = ||I||^2 = 4

    W_batched = jnp.stack([W0, W1])

    params = {"Dense_0": {"kernel": W_batched}}

    loss, metrics = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=False)

    # Expected: Loss(W0) + Loss(W1) = 0 + 4 = 4
    assert jnp.allclose(loss, 4.0, atol=1e-5)


if __name__ == "__main__":
    # Core function tests
    test_adaptive_gram_loss_orthogonal()
    test_adaptive_gram_loss_wide()
    test_adaptive_gram_loss_tall()
    test_adaptive_gram_loss_non_orthogonal()
    test_get_dense_kernels()
    test_get_dense_kernels_with_filter()
    test_find_output_layer_path_by_pattern()
    test_find_output_layer_path_by_index()

    # Loss mode tests
    test_compute_gram_regularization_loss()
    test_compute_gram_regularization_loss_exclude_output()
    test_legacy_api_compute_ortho_loss()

    # Optimizer mode tests
    test_apply_ortho_update_reduces_loss()
    test_apply_ortho_update_preserves_orthogonal()
    test_apply_ortho_update_excludes_output()
    test_apply_ortho_update_batched()

    # PPO integration tests
    test_ppo_with_ortho_loss_mode()
    test_ppo_with_ortho_optimizer_mode()
    test_ppo_with_groupsort_activation()
    test_ppo_train_with_ortho_loss()
    test_ppo_train_with_ortho_optimizer()

    # Backward compatibility tests
    test_ortho_loss_zero_for_orthogonal()
    test_ortho_loss_wide()
    test_exclude_last_layer()
    test_ortho_loss_batched()

    print("All tests passed!")
