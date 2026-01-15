import jax
import wandb
import jax.numpy as jnp
import numpy as np
import jax.tree_util
from rejax import TD3
import datetime
import argparse
import time
from tensorboardX import SummaryWriter
import os
from typing import Any

# Global registry to store loggers, avoiding serialization issues
_LOGGER_REGISTRY = {}

@jax.tree_util.register_pytree_node_class
class ProxyLogger:
    """
    A lightweight proxy for the BatchLogger.
    """
    def __init__(self, logger_id):
        self.logger_id = logger_id

    def log(self, data, step, agent_id):
        # Retrieve the actual logger from registry
        def _log_callback(data, step, agent_id):
             logger = _LOGGER_REGISTRY.get(self.logger_id)
             if logger:
                 logger.log(data, step, agent_id)

        jax.debug.callback(_log_callback, data, step, agent_id)

    def tree_flatten(self):
        return (), (self.logger_id,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

class BatchLogger:
    """
    Logger for vmapped training with ID-based attribution.
    """
    def __init__(self, log_dir, num_seeds, print_freq=1000, wandb_run=None, use_tb=True, use_wandb=True):
        self.writers = []
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_seeds = num_seeds
        self.print_freq = print_freq
        self.wandb_run = wandb_run

        if self.use_tb:
            for i in range(num_seeds):
                seed_dir = os.path.join(log_dir, f"seed_{i}")
                self.writers.append(SummaryWriter(seed_dir))
            
    def log(self, data, step, agent_id):
        # We must handle inputs that might be batched (arrays) OR unbatched (scalars).
        # We assume standard numpy/jax behavior.
        
        step_np = np.array(step)
        id_np = np.array(agent_id)
        
        # If inputs are scalars (0-d), wrap them to 1-d for uniform processing
        if step_np.ndim == 0:
            step_np = step_np[None]
        if id_np.ndim == 0:
            id_np = id_np[None]
        
        # Determine how many items we received (1 if unbatched/loop, N if batched)
        num_items = step_np.shape[0]
        assert id_np.shape[0] == num_items, f"Shape mismatch: step {step_np.shape} vs id {id_np.shape}"

        # Current global step (roughly same for all in batch usually)
        current_step = int(step_np[0])

        # Prepare for aggregation
        agg_values = {} 

        for k, v in data.items():
            v_np = np.array(v)
            if v_np.ndim == 0:
                v_np = np.full((num_items,), v_np) # Broadcast scalar metric
            
            # Store for WandB aggregation later
            agg_values[k] = v_np

            if self.use_tb:
                for i in range(num_items):
                    aid = int(id_np[i])
                    val = float(v_np[i])
                    
                    # Sanity check agent_id
                    if aid < 0 or aid >= self.num_seeds:
                        continue # Should not happen

                    # Filter expensive stats placeholders
                    if ("s_max" in k or "s_min" in k) and val == 0.0:
                        continue
                        
                    self.writers[aid].add_scalar(k, val, current_step)

        # Log Mean and Std to WandB
        if self.use_wandb and self.wandb_run:
            agg_data = {"global_step": current_step}
            for k, v_np in agg_values.items():
                # Filter zeros for spectral stats
                mask = np.ones_like(v_np, dtype=bool)
                if "s_max" in k or "s_min" in k:
                    mask = (v_np != 0.0)

                if mask.any():
                    filtered = v_np[mask]
                    agg_data[f"{k}/mean"] = np.mean(filtered)
                    if len(filtered) > 1:
                        agg_data[f"{k}/std"] = np.std(filtered)

            self.wandb_run.log(agg_data)

        if current_step % self.print_freq == 0:
             # Print critic loss for the first item in batch (or average?)
             avg_loss = np.mean(agg_values.get('critic/loss', [0.0]))
             print(f"Step {current_step}: Critic Loss={avg_loss:.4f}")

    def close(self):
        for w in self.writers:
            w.close()

def run_experiment(config, num_seeds, base_log_dir, use_tb, use_wandb):
    # Determine tags and names
    ortho_method = "gram" if config["ortho_lambda"] > 0 else "none"
    activation = config["actor_kwargs"].get("activation", "swish")
    
    depth = len(config["actor_kwargs"]["hidden_layer_sizes"])
    width = config["actor_kwargs"]["hidden_layer_sizes"][0] if depth > 0 else 0
    
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    group_name = f"{config['algo']}_{config['env']}_{ortho_method}_λ{config['ortho_lambda']}_{activation}_d{depth}_w{width}"
    run_name = f"{group_name}_seeds{num_seeds}_{timestamp}"

    # Verify log dir
    log_dir = os.path.join(base_log_dir, group_name, timestamp)
    if use_tb:
        print(f"Logging {num_seeds} seeds to TensorBoard at: {log_dir}")
    
    wandb_run = None
    if use_wandb:
        print(f"Logging aggregate metrics to WandB run: {run_name}")
        wandb_run = wandb.init(
            project="rejax-plasticity",
            name=run_name,
            group=group_name,
            config={**config, "num_seeds": num_seeds},
            tags=[config["algo"], activation, ortho_method, config['env']],
            mode="online"
        )

    # Setup Vectorized Logger
    logger_id = id(group_name) + int(time.time())
    real_logger = BatchLogger(log_dir, num_seeds, wandb_run=wandb_run, use_tb=use_tb, use_wandb=use_wandb)
    _LOGGER_REGISTRY[logger_id] = real_logger
    proxy_logger = ProxyLogger(logger_id)
    
    # Create the agent template
    # Create the agent template
    # We create ONE agent template, then replicate properties to create a BATCH of agents manually
    for key in ["algo", "seed", "global_seed"]: config.pop(key, None)
    
    # Create base instance (prototype)
    base_algo = TD3.create(**config)
    
    # Define callback once
    old_eval_callback = base_algo.eval_callback
    
    def vectorized_eval_callback(algo, train_state, rng):
        lengths, returns = old_eval_callback(algo, train_state, rng)
        
        def log_eval(step, len_mean, ret_mean, aid):
            real_logger.log({
                "eval/episode_length": len_mean, 
                "eval/return": ret_mean
            }, step, aid)
        
        jax.experimental.io_callback(
            log_eval, (), train_state.global_step, lengths.mean(), returns.mean(), algo.agent_id
        )
        return lengths, returns

    # Apply common modifications to prototype
    base_algo = base_algo.replace(logger=proxy_logger, eval_callback=vectorized_eval_callback)

    # Create batch by replacing only the agent_id
    agents = [base_algo.replace(agent_id=i) for i in range(num_seeds)]
        
    # Stack agents into a single PyTree batch
    # This is effectively "vmap" over the agent initialization
    agent_batch = jax.tree_map(lambda *args: jnp.stack(args), *agents)

    # Vectorized Training!
    print(f"Starting Vectorized Study: {run_name}")
    rng = jax.random.PRNGKey(config.get("global_seed", 0))
    keys = jax.random.split(rng, num_seeds)
    
    # VMAP the train function
    # Map over agent_batch (0) and keys (0)
    vmap_train = jax.jit(jax.vmap(TD3.train, in_axes=(0, 0)))
    
    start_time = time.time()
    ts, _ = vmap_train(agent_batch, keys)
    jax.block_until_ready(ts)
    
    real_logger.close()
    
    duration = time.time() - start_time
    print(f"Finished {run_name} in {duration:.1f}s!")
    
    if use_wandb:
        wandb.log({"time/total_duration": duration, "time/sps": (config["total_timesteps"] * num_seeds) / duration})
        wandb.finish()
        
    del _LOGGER_REGISTRY[logger_id]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="TD3")
    parser.add_argument("--ortho_lambda", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="groupsort")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--total_timesteps", type=int, default=50_000)
    parser.add_argument("--log_expensive_freq", type=int, default=500)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--envs", type=str, nargs="+", default=["brax/ant", "brax/halfcheetah", "brax/humanoid", "brax/walker2d"])
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--logging", type=str, default="both", choices=["wandb", "tensorboard", "both"])
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs per seed (GPU utilization)")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    BASE_CONFIG = {
        "algo": args.algo,
        "env_params": {},
        "eval_freq": 2000,
        "num_envs": args.num_envs,
        "learning_rate": 3e-4,
        "total_timesteps": args.total_timesteps,
        "fill_buffer": 1000,
        "buffer_size": 100_000,
        "batch_size": args.batch_size,
        "gamma": 0.99,
        "exploration_noise": 0.1,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,
        "policy_delay": 2,
        "ortho_lambda": args.ortho_lambda,
        "log_expensive_freq": args.log_expensive_freq,
        "global_seed": args.global_seed,
    }
    
    use_tb = args.logging in ["tensorboard", "both"]
    use_wandb = args.logging in ["wandb", "both"]

    for env_name in args.envs:
        for depth in args.depths:
            config = BASE_CONFIG.copy()
            config["env"] = env_name
            config["actor_kwargs"] = {"activation": args.activation, "hidden_layer_sizes": (args.width,) * depth}
            config["critic_kwargs"] = {"activation": args.activation, "hidden_layer_sizes": (args.width,) * depth}
            run_experiment(config, args.num_seeds, args.log_dir, use_tb, use_wandb)
