import brax_test_envs
from brax import envs
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

env = envs.get_environment(env_name='first', backend='position')
rng = jax.random.PRNGKey(0)
num_envs = 4
keys = jax.random.split(rng, num_envs)
env = envs.training.wrap(
            env,
            episode_length=1,
            action_repeat=1,
        )
env_state = env.reset(keys)
# breakpoint()
rollout = {'obs':[], 'reward':[], 'done': []}
for i in range(10):
    for k in rollout.keys():
        rollout[k].append(getattr(env_state, k))
    env_state = env.step(env_state, jnp.array([1.]*num_envs, dtype=jnp.float32).reshape(4,1))
    breakpoint()

import matplotlib.pyplot as plt
import numpy as np

# Extract data from rollout
obs_data = np.array(rollout['obs'])  # Shape: (time_steps, num_envs, obs_dim)
reward_data = np.array(rollout['reward'])  # Shape: (time_steps, num_envs)
done_data = np.array(rollout['done'])  # Shape: (time_steps, num_envs)

time_steps = range(len(rollout['obs']))

# Plot a single observation dimension for all environments
plt.figure(figsize=(10, 6))
for env_idx in range(obs_data.shape[1]):
    plt.plot(time_steps, obs_data[:, env_idx, 0], label=f'Env {env_idx}')  # Assume dimension 0
plt.title('Observation (First Dimension) Across Environments')
plt.xlabel('Time Step')
plt.ylabel('Observation Value')
plt.legend()
plt.grid(True)
plt.savefig('observation.jpg')

# Plot rewards for all environments in a single plot
plt.figure(figsize=(10, 6))
for env_idx in range(reward_data.shape[1]):
    plt.plot(time_steps, reward_data[:, env_idx], label=f'Env {env_idx}')
plt.title('Rewards Across Environments')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.savefig('reward.jpg')

# Plot done flags for all environments in a single plot
plt.figure(figsize=(10, 6))
for env_idx in range(done_data.shape[1]):
    plt.plot(time_steps, done_data[:, env_idx], label=f'Env {env_idx}', marker='o')
plt.title('Done Flags Across Environments')
plt.xlabel('Time Step')
plt.ylabel('Done (True/False)')
plt.legend()
plt.grid(True)
plt.savefig('done.jpg')

