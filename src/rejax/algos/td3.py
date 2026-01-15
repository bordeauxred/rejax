from typing import Any
import chex
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks import DeterministicPolicy, QNetwork


# Algorithm outline
# num_eval_iterations = total_timesteps / eval_freq
# num_train_iterations = eval_freq / (num_envs * policy_delay)
# for _ in range(num_eval_iterations):
#   for _ in range(num_train_iterations):
#     for _ in range(policy_delay):
#       M = collect num_gradient_steps minibatches
#       update critic using M
#     update actor using M
#     update target networks


class TD3(
    ReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    Algorithm,
):
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_critics: int = struct.field(pytree_node=False, default=2)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    exploration_noise: chex.Scalar = struct.field(pytree_node=True, default=0.3)
    target_noise: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    target_noise_clip: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    policy_delay: int = struct.field(pytree_node=False, default=2)
    ortho_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.0)
    log_expensive_freq: int = struct.field(pytree_node=False, default=5000)
    logger: Any = struct.field(pytree_node=False, default=None)
    agent_id: int = struct.field(pytree_node=True, default=0)

    def make_act(self, ts):
        def act(obs, rng):
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.actor.apply(ts.actor_ts.params, obs)
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        actor_kwargs = config.pop("actor_kwargs", {})
        activation = actor_kwargs.pop("activation", "swish")
        if activation == "groupsort":
            from rejax.networks import groupsort
            actor_kwargs["activation"] = groupsort
        else:
            actor_kwargs["activation"] = getattr(nn, activation)
        action_range = (
            env.action_space(env_params).low,
            env.action_space(env_params).high,
        )
        action_dim = np.prod(env.action_space(env_params).shape)
        hidden_layer_sizes = actor_kwargs.pop("hidden_layer_sizes", (64, 64))
        actor = DeterministicPolicy(
            action_dim, action_range, hidden_layer_sizes=hidden_layer_sizes, **actor_kwargs
        )

        critic_kwargs = config.pop("critic_kwargs", {})
        if activation == "groupsort":
            from rejax.networks import groupsort
            critic_kwargs["activation"] = groupsort
        else:
            critic_kwargs["activation"] = getattr(nn, activation)
        hidden_layer_sizes = critic_kwargs.pop("hidden_layer_sizes", (64, 64))
        critic = QNetwork(hidden_layer_sizes=hidden_layer_sizes, **critic_kwargs)

        return {"actor": actor, "critic": critic}

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        rng_critic = jax.random.split(rng_critic, self.num_critics)
        obs_ph = jnp.empty((1, *self.env.observation_space(self.env_params).shape))
        action_ph = jnp.empty((1, *self.env.action_space(self.env_params).shape))

        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        actor_params = self.actor.init(rng_actor, obs_ph)
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)

        vmap_init = jax.vmap(self.critic.init, in_axes=(0, None, None))
        critic_params = vmap_init(rng_critic, obs_ph, action_ph)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)
        return {
            "actor_ts": actor_ts,
            "actor_target_params": actor_params,
            "critic_ts": critic_ts,
            "critic_target_params": critic_params,
        }

    @property
    def vmap_critic(self):
        return jax.vmap(self.critic.apply, in_axes=(0, None, None))

    def train(self, rng=None, train_state=None):
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)

        def eval_iteration(ts, unused):
            # Run a few trainig iterations
            steps_per_train_it = self.num_envs * self.policy_delay
            num_train_its = np.ceil(self.eval_freq / steps_per_train_it).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_train_its,
                lambda _, ts: self.train_iteration(ts),
                ts,
            )

            # Run evaluation
            return ts, self.eval_callback(self, ts, ts.rng)

        ts, evaluation = jax.lax.scan(
            eval_iteration,
            ts,
            None,
            np.ceil(self.total_timesteps / self.eval_freq).astype(int),
        )

        if not self.skip_initial_evaluation:
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation

    def train_iteration(self, ts):
        old_global_step = ts.global_step
        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )
        ts, minibatch, critic_metrics = jax.lax.fori_loop(
            0,
            self.policy_delay,
            lambda _, val: (self.train_critic(val[0])[0], val[1], val[2]), # Dropping extra critic metrics from inner loop?
            # Wait, `train_critic` returns (ts, minibatches, metrics).
            # We need to accumulate metrics? Or just take the last one?
            # Creating a robust logging pipeline inside `scan/fori_loop` is tricky.
            # Let's simplify: `train_critic` runs one epoch of updates.
            # We call it `policy_delay` times.
            # We only care about logging, so averaging over these steps is fine.
            
            # Actually, `train_critic` gathers `num_epochs` metrics.
            # The fori_loop runs `policy_delay` times.
            # Accessing the metrics from inside fori_loop is hard.
            # We will refactor to use `scan`.
            
            (ts, placeholder_minibatch, None), # Initial state
        )
        # Scan version:
        def train_critic_scan(carry, _):
            ts = carry
            ts, mbs, metrics = self.train_critic(ts)
            return ts, (mbs, metrics)
            
        ts, (minibatches_stack, critic_metrics_stack) = jax.lax.scan(
            train_critic_scan,
            ts,
            None,
            self.policy_delay
        )
        # minibatches_stack: (policy_delay, num_epochs, batch_size, ...)
        # We need to flatten to pass to train_policy?
        # `train_policy` expects (num_epochs, batch_size, ...)
        # But `train_critic` generates new minibatches every time it's called?
        # Yes, `collect_transitions`.
        
        # We only need the minibatches from the LAST `train_critic` call to update the actor?
        # Standard TD3: Update Actor once every d steps.
        # Update Critic d times.
        # Usually Actor uses the same batch as the last Critic update? 
        # Or a fresh batch?
        # Standard: "Update the actor policy ... using the mini-batch sampled for the critic update"
        
        # So we take the last minibatch sequence.
        minibatch = jax.tree.map(lambda x: x[-1], minibatches_stack)
        
        ts, actor_metrics = self.train_policy(ts, minibatch, old_global_step)
        
        # Combine metrics for logging (Restored)
        def mean_metric(m):
             return jnp.mean(m)
             
        critic_metrics_flat = jax.tree.map(mean_metric, critic_metrics_stack)
        actor_metrics_flat = jax.tree.map(mean_metric, actor_metrics)
        
        # Use custom logger if provided, else default to wandb
        def log_metrics(step, c_met, a_met, agent_id):
             # Flatten dicts
             c_flat = {f"critic/{k}": v for k,v in flatten_dict(c_met, sep="/").items()}
             a_flat = {f"actor/{k}": v for k,v in flatten_dict(a_met, sep="/").items()}

             log_data = {**c_flat, **a_flat}
             # For vmap compatibility, we avoid checking values of array-valued metrics here.
             # Filtering must happen in the logger.

             if self.logger is not None:
                 # Pass to external logger (e.g. for vmapped aggregation)
                 self.logger.log(log_data, step, agent_id)
                 return

             # Default WandB logging (only if wandb is initialized)
             log_data["global_step"] = step
             try:
                 import wandb
                 if wandb.run is not None:
                     wandb.log(log_data)
             except (ImportError, Exception):
                 pass
                 
             if step % 1000 == 0:
                 print(f"Step {step}: Critic Loss={log_data.get('critic/loss', 0.0):.4f}, Actor Loss={log_data.get('actor/loss', 0.0):.4f}")

        # Import wandb safely?
        # Ideally we pass a callback function.
        # But for now, we assume `wandb` is globally available or use `host_callback`.
        # `rejax` puts the callback on the algo object. 
        
        import wandb
        
        jax.experimental.io_callback(
            log_metrics,
            (),
            ts.global_step,
            critic_metrics_flat,
            actor_metrics_flat,
            self.agent_id
        )

        return ts

    def train_critic(self, ts):
        start_training = ts.global_step > self.fill_buffer

        # Collect transition
        uniform = jnp.logical_not(start_training)
        ts, transitions = self.collect_transitions(ts, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

        def update_iteration(ts, unused):
            # Sample minibatch
            rng, rng_sample = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            minibatch = ts.replay_buffer.sample(self.batch_size, rng_sample)
            if self.normalize_observations:
                minibatch = minibatch._replace(
                    obs=self.normalize_obs(ts.obs_rms_state, minibatch.obs),
                    next_obs=self.normalize_obs(ts.obs_rms_state, minibatch.next_obs),
                )
            if self.normalize_rewards:
                minibatch = minibatch._replace(
                    reward=self.normalize_rew(ts.rew_rms_state, minibatch.reward)
                )

            # Update network
            # Log expensive stats periodically
            log_expensive = (ts.global_step % self.log_expensive_freq == 0)
            
            ts, metrics = self.update_critic(ts, minibatch, do_expensive_logging=log_expensive)
            return ts, (minibatch, metrics)

        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )

        def do_updates(ts):
            ts, (minibatches, metrics) = jax.lax.scan(update_iteration, ts, None, self.num_epochs)
            return ts, minibatches, metrics

        # Helper to create placeholder metrics
        def get_zero_metrics(ts, mb):
            # Run one dummy update to discover shape
            _, dummy_metrics = self.update_critic(ts, mb, do_expensive_logging=False)
            
            # Create stacked zeros
            return jax.tree.map(
                lambda x: jnp.zeros((self.num_epochs,) + x.shape, x.dtype),
                dummy_metrics
            )
            
        def no_updates(ts):
            mb0 = jax.tree.map(lambda x: x[0], placeholder_minibatch)
            zero_metrics = get_zero_metrics(ts, mb0)
            return ts, placeholder_minibatch, zero_metrics

        ts, minibatches, metrics = jax.lax.cond(
            start_training,
            do_updates,
            no_updates,
            ts,
        )
        
        return ts, minibatches, metrics

    def train_policy(self, ts, minibatches, old_global_step):
        def do_updates(ts):
            ts, metrics = jax.lax.scan(
                lambda ts, minibatch: self.update_actor(ts, minibatch),
                ts,
                minibatches,
            )
            return ts, metrics

        start_training = ts.global_step > self.fill_buffer

        # We need to handling the metrics return when not training
        # We can't easily return valid metrics matching the structure if we don't run.
        # So we run `do_updates` but masking the update effect? No, that's expensive.
        # We will use the same pattern as train_critic:
        # Create a placeholder structure for metrics.
        
        # To get the structure, we can trace a dummy call or use `eval_shape`.
        # Or simplistic approach: We just rely on JAX cond to return zeros.
        
        # But wait, `train_policy` is called unconditionally inside `train_iteration`.
        # If start_training is false, we just return ts.
        # We need to return metrics.
        
        # Let's define the scan function properly to return metrics.
        
        def no_updates(ts):
             # Return dummy metrics
             # We need to know the shape/structure of actor_metrics.
             # We can cheat by running one step on dummy data and discarding it?
             # Or constructing it manually.
             
             # BETTER: run update_actor on the first minibatch but wrapped in "stop_gradient" 
             # and discarding params update? Too complex.
             
             # Best approach for JAX: use `jax.eval_shape` or similar if possible.
             # Or just hardcode the zero-structure if we know it. 
             # But we are adding dynamic metrics (ortho).
             
             # Actually, if we just run `update_actor` with zero-learning rate optimizer?
             # No, simply use `jax.tree_map(jnp.zeros_like, ...)` on the output of a real call?
             
             # Let's execute one real call to get structure (JIT compiles it out if unused?)
             # No, that's risky.
             
             # Let's rely on `jax.lax.cond` doing shape inference. 
             # We need a dummy input for the "false" branch.
             # This is annoying in JAX.
             
             # Simplification: `train_policy` will always return `metrics`.
             # If `start_training` is False, we return a tree of NaNs or Zeros.
             # We can get the structure by running `update_actor` on a dummy batch inside a `jax.eval_shape`?
             # `jax.eval_shape` is purely abstract.
             
             # Let's try to pass `do_updates` result structure out. 
             
             # Actually, simpler pattern:
             # Run `update_actor` but mix the new params with old params based on `start_training`.
             # i.e. always compute everything, but `ts = select(start_training, new_ts, old_ts)`.
             # This wastes compute before `fill_buffer` (random actions period).
             # But `fill_buffer` is short (1000 steps).
             # Is it okay? Maybe. 
             
             # Wait, `update_actor` is expensive (gradients).
             # We should avoid running it if `start_training` is False.
             
             # Let's use the provided `minibatches`. The first epoch's batch.
             mb0 = jax.tree.map(lambda x: x[0], minibatches)
             
             dummy_ts, dummy_metrics = self.update_actor(ts, mb0, do_expensive_logging=False)
             zero_metrics = jax.tree.map(jnp.zeros_like, dummy_metrics)
             
             # scan returns stacked metrics (num_epochs, ...)
             stacked_zero_metrics = jax.tree.map(
                 lambda x: jnp.zeros((self.num_epochs,) + x.shape, x.dtype),
                 dummy_metrics
             )
             return ts, stacked_zero_metrics

        ts, metrics = jax.lax.cond(start_training, do_updates, no_updates, ts)

        # Update target networks
        if self.target_update_freq == 1:
            critic_tp = self.polyak_update(ts.critic_ts.params, ts.critic_target_params)
            actor_tp = self.polyak_update(ts.actor_ts.params, ts.actor_target_params)
        else:
            update_target_params = (
                ts.global_step % self.target_update_freq
                <= old_global_step % self.target_update_freq
            )
            critic_tp = jax.tree.map(
                lambda q, qt: jax.lax.select(update_target_params, q, qt),
                self.polyak_update(ts.critic_ts.params, ts.critic_target_params),
                ts.critic_target_params,
            )
            actor_tp = jax.tree.map(
                lambda pi, pit: jax.lax.select(update_target_params, pi, pit),
                self.polyak_update(ts.actor_ts.params, ts.actor_target_params),
                ts.actor_target_params,
            )

        ts = ts.replace(critic_target_params=critic_tp, actor_target_params=actor_tp)
        return ts, metrics

    def collect_transitions(self, ts, uniform=False):
        # Sample actions
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_uniform(rng):
            sample_fn = self.env.action_space(self.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, self.num_envs))

        def sample_policy(rng):
            if self.normalize_observations:
                last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            actions = self.actor.apply(ts.actor_ts.params, last_obs)
            noise = self.exploration_noise * jax.random.normal(rng, actions.shape)
            action_low, action_high = self.action_space.low, self.action_space.high
            return jnp.clip(actions + noise, action_low, action_high)

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        # Step environment
        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, actions, self.env_params
        )

        if self.normalize_observations:
            ts = ts.replace(
                obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs)
            )
        if self.normalize_rewards:
            ts = ts.replace(
                rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones)
            )

        # Return minibatch and updated train state
        minibatch = Minibatch(
            obs=ts.last_obs,
            action=actions,
            reward=rewards,
            next_obs=next_obs,
            done=dones,
        )
        ts = ts.replace(
            last_obs=next_obs,
            env_state=env_state,
            global_step=ts.global_step + self.num_envs,
        )
        return ts, minibatch

    def update_critic(self, ts, minibatch, do_expensive_logging=False):
        def critic_loss_fn(params):
            action = self.actor.apply(ts.actor_target_params, minibatch.next_obs)
            noise = jnp.clip(
                self.target_noise * jax.random.normal(ts.rng, action.shape),
                -self.target_noise_clip,
                self.target_noise_clip,
            )
            action_low, action_high = self.action_space.low, self.action_space.high
            action = jnp.clip(action + noise, action_low, action_high)

            qs_target = self.vmap_critic(
                ts.critic_target_params, minibatch.next_obs, action
            )
            q_target = jnp.min(qs_target, axis=0)
            target = minibatch.reward + (1 - minibatch.done) * self.gamma * q_target
            q1, q2 = self.vmap_critic(params, minibatch.obs, minibatch.action)

            loss_q1 = optax.l2_loss(q1, target).mean()
            loss_q2 = optax.l2_loss(q2, target).mean()
            
            # Add ortho loss
            from rejax.regularization import compute_ortho_loss
            
            ortho_loss, ortho_metrics = compute_ortho_loss(
                params, self.ortho_lambda, log_now=do_expensive_logging
            )
            
            total_loss = loss_q1 + loss_q2 + ortho_loss
            
            # Metrics
            metrics = {
                "loss": total_loss,
                "qf1_loss": loss_q1,
                "qf2_loss": loss_q2,
                "ortho_loss": ortho_loss,
                "q_values_mean": q1.mean(),
                "q_values_std": q1.std(),
                "targets_mean": target.mean(),
                "bellman_error": ((q1 - target)**2).mean(), # MSBE (same as loss_q1)
            }
            
            # Explained Variance
            # var(y - pred) / var(y)
            y_var = jnp.var(target)
            diff_var = jnp.var(target - q1)
            metrics["explained_variance"] = 1.0 - diff_var / (y_var + 1e-8)

            metrics.update(ortho_metrics)
            return total_loss, metrics

        (loss, metrics), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(ts.critic_ts.params)
        
        # Gradient Norms
        metrics["grad_norm"] = optax.global_norm(grads)
        flat_grads = flatten_dict(grads, sep="/")
        for k, v in flat_grads.items():
            if k.endswith("/kernel"):
                layer_name = "_".join(k.split("/")[:-1])
                metrics[f"diag/grad_norm_{layer_name}"] = jnp.linalg.norm(v)
        
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts, metrics

    def update_actor(self, ts, minibatch, do_expensive_logging=False):
        def actor_loss_fn(params):
            action = self.actor.apply(params, minibatch.obs)
            q = self.vmap_critic(ts.critic_ts.params, minibatch.obs, action)
            
            # Add ortho loss
            from rejax.regularization import compute_ortho_loss
            
            ortho_loss, ortho_metrics = compute_ortho_loss(
                params, self.ortho_lambda, log_now=do_expensive_logging
            )

            total_loss = -q.mean() + ortho_loss
            
            metrics = {
                "loss": total_loss,
                "q_val": q.mean(),
                "ortho_loss": ortho_loss
            }
            metrics.update(ortho_metrics)
            return total_loss, metrics

        (loss, metrics), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        
        # Gradient Norms
        metrics["grad_norm"] = optax.global_norm(grads)
        flat_grads = flatten_dict(grads, sep="/")
        for k, v in flat_grads.items():
            if k.endswith("/kernel"):
                layer_name = "_".join(k.split("/")[:-1])
                metrics[f"diag/grad_norm_{layer_name}"] = jnp.linalg.norm(v)

        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, metrics
