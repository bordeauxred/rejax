import jax
import jax.numpy as jnp
from rejax.regularization import compute_ortho_loss

def test_ortho_loss_zero_for_orthogonal():
    # Construct an orthogonal matrix
    key = jax.random.PRNGKey(0)
    W = jax.random.orthogonal(key, 4) # 4x4
    
    params = {"Dense_0": {"kernel": W}}
    
    # Square matrix: W.T @ W = I
    # The loss should be close to 0
    loss, metrics = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=False)
    
    assert jnp.allclose(loss, 0.0, atol=1e-5)

def test_ortho_loss_wide():
    # Wide matrix: 2 rows, 4 cols
    # Should enforce row orthogonality: W @ W.T = (n_out/n_in) * I ??
    # Wait, my implementation said:
    # if n_in < n_out: # Wide
    #    gram = W @ W.T
    #    scale = n_out / n_in
    #    identity = scale * I
    
    # Let's test this scaling.
    # W = [[1, 0, 1, 0], [0, 1, 0, 1]] -> 2x4
    # W @ W.T = [[2, 0], [0, 2]]
    # n_in=2, n_out=4 -> scale = 4/2 = 2.
    # identity = 2 * I = [[2, 0], [0, 2]]
    # So this matrix should have 0 loss.
    
    W = jnp.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    params = {"Dense_0": {"kernel": W}}
    
    loss, metrics = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=False)
    assert jnp.allclose(loss, 0.0, atol=1e-5)

def test_exclude_last_layer():
    # Setup params that mimic a 2-layer MLP
    # Dense_0 (hidden)
    # Dense_1 (output)
    
    W0 = jnp.ones((2, 2)) # High loss
    W1 = jnp.ones((2, 2)) # High loss
    
    params = {
        "Dense_0": {"kernel": W0},
        "Dense_1": {"kernel": W1}
    }
    
    # If exclude_last_layer=True, loss should come ONLY from W0.
    loss_excl, _ = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=True)
    
    # If exclude_last_layer=False, loss should come from W0 + W1.
    loss_all, _ = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=False)
    
    # Loss for W0:
    # W0 = [[1,1],[1,1]]. W0.T @ W = [[2,2],[2,2]]. I = [[1,0],[0,1]].
    # Diff = [[1,2],[2,1]]. Sq = [[1,4],[4,1]]. Sum = 10.
    expected_loss_W0 = 10.0
    
    # Total loss should be 20.0
    assert jnp.allclose(loss_excl, 10.0, atol=1e-5)
    assert jnp.allclose(loss_all, 20.0, atol=1e-5)

def test_ortho_loss_batched():
    # Simulate ensemble of 2 critics
    # Critic 0: Orthogonal (loss=0)
    # Critic 1: Non-orthogonal (loss>0)
    
    W0 = jax.random.orthogonal(jax.random.PRNGKey(0), 4)
    W1 = jnp.array([[1., 0., 1., 0.], [0., 1., 0., 1.]]).T # Tall: 4x2
    # W1.T @ W1 = [[2, 0], [0, 2]]. I = [[1, 0], [0, 1]]. 
    # Diff = [[1, 0], [0, 1]]. Sq = [[1, 0], [0, 1]]. Sum = 2.
    
    # W1 should have specific loss.
    # We want W^T @ W - I = I, so that Loss = Frobenius(I)^2 = 4.
    # This implies W^T @ W = 2I.
    # So W should be sqrt(2) * I.
    import numpy as np
    W1 = jnp.eye(4) * np.sqrt(2) 
    
    W_batched = jnp.stack([W0, W1]) # Shape (2, 4, 4)
    
    params = {"Dense_0": {"kernel": W_batched}}
    
    loss, metrics = compute_ortho_loss(params, lambda_coeff=1.0, exclude_last_layer=False)
    
    # Expected loss = Loss(W0) + Loss(W1) = 0 + 4 = 4.
    assert jnp.allclose(loss, 4.0, atol=1e-5)

if __name__ == "__main__":
    test_ortho_loss_zero_for_orthogonal()
    test_ortho_loss_wide()
    test_exclude_last_layer()
    test_ortho_loss_batched()
    print("All tests passed!")
