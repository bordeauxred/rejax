import jax
import jax.numpy as jnp
from rejax.networks import groupsort

def test_groupsort_basic():
    x = jnp.array([4.0, 1.0, 3.0, 2.0, 6.0, 5.0])
    
    # group_size = 2
    # Expected: [1.0, 4.0, 2.0, 3.0, 5.0, 6.0]
    expected_2 = jnp.array([1.0, 4.0, 2.0, 3.0, 5.0, 6.0])
    out_2 = groupsort(x, group_size=2)
    assert jnp.allclose(out_2, expected_2)
    
    # group_size = 3
    # Expected: [1.0, 3.0, 4.0, 2.0, 5.0, 6.0]
    expected_3 = jnp.array([1.0, 3.0, 4.0, 2.0, 5.0, 6.0])
    out_3 = groupsort(x, group_size=3)
    assert jnp.allclose(out_3, expected_3)

def test_groupsort_batch():
    x = jnp.array([
        [4.0, 1.0, 3.0, 2.0],
        [8.0, 7.0, 6.0, 5.0]
    ])
    
    # group_size = 2
    # Expected:
    # [[1.0, 4.0, 2.0, 3.0],
    #  [7.0, 8.0, 5.0, 6.0]]
    expected = jnp.array([
        [1.0, 4.0, 2.0, 3.0],
        [7.0, 8.0, 5.0, 6.0]
    ])
    out = groupsort(x, group_size=2)
    assert jnp.allclose(out, expected)

def test_groupsort_jit():
    x = jnp.array([4.0, 1.0, 3.0, 2.0])
    
    @jax.jit
    def fast_groupsort(x):
        return groupsort(x, group_size=2)
    
    expected = jnp.array([1.0, 4.0, 2.0, 3.0])
    out = fast_groupsort(x)
    assert jnp.allclose(out, expected)

if __name__ == "__main__":
    test_groupsort_basic()
    test_groupsort_batch()
    test_groupsort_jit()
    print("All tests passed!")
