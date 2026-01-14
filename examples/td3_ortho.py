import jax
import wandb
import jax.numpy as jnp
from rejax import TD3

if __name__ == "__main__":
    # Initialize WandB
    CONFIG = {
        "env": "Pendulum-v1",
        "env_params": {},
        "eval_freq": 2000,
        "num_envs": 1,
        "learning_rate": 3e-4,
        "total_timesteps": 50_000,
        "fill_buffer": 1000,
        "buffer_size": 100_000,
        "batch_size": 256,
        "gamma": 0.99,
        # TD3 specific parameters
        "exploration_noise": 0.1,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,
        "policy_delay": 2,
        # Orthogonalization Regularization
        "ortho_lambda": 0.2,
        "log_expensive_freq": 500,
        # Network parameters with GroupSort
        "actor_kwargs": {"activation": "groupsort", "hidden_layer_sizes": (64, 64)},
        "critic_kwargs": {"activation": "groupsort", "hidden_layer_sizes": (64, 64)},
    }
    
    wandb.init(project="rejax-plasticity", config=CONFIG, tags=["td3", "groupsort", "ortho"], mode="online")
    
    # Create the agent
    print("Initializing TD3 with GroupSort and Orthogonalization...")
    # rejax's .create() does not automatically pass unknown kwargs to the class 
    # unless they are defined fields. TD3 fields need to include ortho_lambda.
    # I added ortho_lambda to TD3 class definition.
    algo = TD3.create(**CONFIG)
    
    # Setup WandB Logging Callback
    old_eval_callback = algo.eval_callback

    def wandb_callback(algo, train_state, rng):
        lengths, returns = old_eval_callback(algo, train_state, rng)
        
        def log(step, len_mean, ret_mean):
            wandb.log({
                "episode_length": len_mean, 
                "return": ret_mean,
                "global_step": step
            })
        
        jax.experimental.io_callback(
            log,
            (),
            train_state.global_step,
            lengths.mean(),
            returns.mean()
        )
        return lengths, returns

    algo = algo.replace(eval_callback=wandb_callback)
    
    print("Training...")
    
    # Train the agent
    rng = jax.random.PRNGKey(0)
    train_fn = jax.jit(algo.train)
    ts, _ = train_fn(rng)
    
    print("Training complete!")
    wandb.finish()
