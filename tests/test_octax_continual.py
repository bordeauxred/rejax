"""
Tests for Octax continual learning benchmark.

Local tests with max 15k steps to keep runtime short.
"""
import pytest
import jax
import jax.numpy as jnp


# Skip all tests if octax is not installed
pytest.importorskip("octax")


class TestOctax2Gymnax:
    """Test the Octax-to-Gymnax wrapper."""

    def test_create_environments(self):
        """Test that all games create successfully."""
        from rejax.compat.octax2gymnax import create_octax

        games = ["brix", "pong", "tetris", "tank", "spacejam", "deep"]
        for game in games:
            env, env_params = create_octax(game)
            assert env is not None
            assert env_params is not None

    def test_observation_shape(self):
        """Verify (64, 32, 4) shape after transpose."""
        from rejax.compat.octax2gymnax import create_octax

        env, env_params = create_octax("brix")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng, env_params)

        # Paper: (4, 64, 32) -> transposed to (64, 32, 4) for CNN
        assert obs.shape == (64, 32, 4), f"Expected (64, 32, 4), got {obs.shape}"
        assert obs.dtype == jnp.float32

    def test_step_observation_shape(self):
        """Verify step also returns correct shape."""
        from rejax.compat.octax2gymnax import create_octax

        env, env_params = create_octax("brix")
        rng = jax.random.PRNGKey(0)
        rng_reset, rng_step = jax.random.split(rng)

        obs, state = env.reset(rng_reset, env_params)
        action = 0  # No-op
        next_obs, next_state, reward, done, info = env.step(rng_step, state, action, env_params)

        assert next_obs.shape == (64, 32, 4)
        assert next_obs.dtype == jnp.float32

    def test_action_spaces(self):
        """Verify action dimensions per game."""
        from rejax.compat.octax2gymnax import create_octax

        expected_actions = {
            "brix": 3,
            "pong": 3,
            "tetris": 5,
            "tank": 6,
            "spacejam": 5,
            "deep": 4,
        }

        for game, expected in expected_actions.items():
            env, env_params = create_octax(game)
            assert env.num_actions == expected, f"{game}: expected {expected}, got {env.num_actions}"


class TestUnifiedOctaxEnv:
    """Test the unified action space wrapper."""

    def test_unified_action_space(self):
        """All games should have same action space (6 = max)."""
        import sys
        sys.path.insert(0, "scripts")
        from bench_octax_continual import create_unified_env, UNIFIED_ACTIONS

        for game in ["brix", "tetris", "tank"]:
            env, env_params = create_unified_env(game)
            assert env.num_actions == UNIFIED_ACTIONS

    def test_action_mapping(self):
        """Invalid actions should map to no-op."""
        import sys
        sys.path.insert(0, "scripts")
        from bench_octax_continual import create_unified_env, UNIFIED_ACTIONS

        # Brix has 3 actions (0, 1, 2), so action 5 should map to 0
        env, env_params = create_unified_env("brix")
        rng = jax.random.PRNGKey(0)
        rng_reset, rng_step = jax.random.split(rng)

        obs, state = env.reset(rng_reset, env_params)

        # Action 5 > 3 (brix actions), should map to 0
        invalid_action = 5
        next_obs, next_state, reward, done, info = env.step(
            rng_step, state, invalid_action, env_params
        )

        # Should not crash, observation should be valid
        assert next_obs.shape == (64, 32, 4)


class TestOctaxPPOConfig:
    """Test PPO configuration for Octax."""

    def test_paper_config_defaults(self):
        """Verify paper defaults are used."""
        import sys
        sys.path.insert(0, "scripts")
        from bench_octax_continual import create_octax_ppo_config, create_unified_env

        env, env_params = create_unified_env("brix")
        config = create_octax_ppo_config(env, env_params, total_timesteps=100000)

        # Paper defaults
        assert config["num_epochs"] == 8, "Paper uses 8 epochs"
        assert config["num_envs"] == 512, "Paper default is 512 envs"
        assert config["num_steps"] == 32, "Paper uses 32 steps"
        assert config["learning_rate"] == 5e-4, "Paper uses lr=5e-4"
        assert config["agent_kwargs"]["mlp_hidden_sizes"] == (256,), "Paper uses single 256-unit layer"


class TestOctaxContinualTrainer:
    """Test the continual learning trainer."""

    @pytest.mark.parametrize("steps", [15000])
    def test_smoke_training(self, steps):
        """Quick training smoke test (max 15k steps)."""
        import sys
        sys.path.insert(0, "scripts")
        from bench_octax_continual import OctaxContinualTrainer, EXPERIMENT_CONFIGS

        trainer = OctaxContinualTrainer(
            config_name="paper_256x1",
            experiment_config=EXPERIMENT_CONFIGS["paper_256x1"],
            steps_per_task=steps,
            num_cycles=1,
            num_envs=64,  # Small for fast test
            eval_freq=steps,
            use_wandb=False,
            task_list=["brix"],  # Single task for speed
        )

        rng = jax.random.PRNGKey(42)
        results = trainer.run(rng)

        assert "per_task_results" in results
        assert len(results["per_task_results"]) == 1
        assert results["per_task_results"][0]["task"] == "brix"


class TestOctaxCNN:
    """Test the OctaxCNN architecture."""

    def test_cnn_output_shape(self):
        """Verify CNN feature extraction produces expected shape."""
        from rejax.networks import OctaxCNN
        import jax.numpy as jnp

        cnn = OctaxCNN(mlp_hidden_sizes=(256,))  # Paper-style single layer
        rng = jax.random.PRNGKey(0)

        # Batch of 4 observations, (64, 32, 4) input
        obs = jnp.ones((4, 64, 32, 4))
        params = cnn.init(rng, obs)
        features = cnn.apply(params, obs)

        # After CNN + MLP(256), should output (batch, 256)
        assert features.shape == (4, 256), f"Expected (4, 256), got {features.shape}"

    def test_cnn_deep_mlp(self):
        """Verify deep MLP variant works."""
        from rejax.networks import OctaxCNN
        import jax.numpy as jnp

        cnn = OctaxCNN(mlp_hidden_sizes=(256, 256, 256, 256))  # Deep MLP
        rng = jax.random.PRNGKey(0)

        obs = jnp.ones((4, 64, 32, 4))
        params = cnn.init(rng, obs)
        features = cnn.apply(params, obs)

        assert features.shape == (4, 256)

    def test_policy_network(self):
        """Test discrete policy with OctaxCNN backbone."""
        from rejax.networks import DiscreteOctaxCNNPolicy
        import jax.numpy as jnp

        policy = DiscreteOctaxCNNPolicy(
            action_dim=6,
            mlp_hidden_sizes=(256,),
        )
        rng = jax.random.PRNGKey(0)
        rng_init, rng_act = jax.random.split(rng)

        obs = jnp.ones((4, 64, 32, 4))
        params = policy.init(rng_init, obs, rng_act)
        action, log_prob, entropy = policy.apply(params, obs, rng_act)

        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
