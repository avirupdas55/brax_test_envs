import enum
from typing import Dict, Any, Mapping, Optional, Tuple, Union
from brax import base
from brax.envs.base import PipelineEnv, State
import jax
import jax.numpy as jnp
from flax import struct

Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]

@struct.dataclass
class ExtendedState(base.Base):
    """State with an additional rng field."""
    pipeline_state: Optional[base.State]
    obs: Observation
    reward: jax.Array
    done: jax.Array
    rng: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

class ObservationMode(enum.Enum):
    """
    Describes observation formats.

    Attributes:
        NDARRAY: Flat numpy array of state information.
        DICT_STATE: Dictionary of state information.
        DICT_PIXELS: Dictionary of pixel observations.
        DICT_PIXELS_STATE: Dictionary of pixel and state information.
    """
    NDARRAY = 'ndarray'
    DICT_STATE = 'dict_state'
    DICT_PIXELS = 'dict_pixels'
    DICT_PIXELS_STATE = 'dict_pixels_state'

class FirstEnv(PipelineEnv):
    """One action, zero observation, one timestep long, +1 reward every timestep."""
    def __init__(
            self,
            obs_mode: ObservationMode = ObservationMode.NDARRAY,
            **kwargs,
    ):
        self._obs_mode = ObservationMode(obs_mode)
        self._step_count = 0

    def reset(self, rng: jax.Array) -> ExtendedState:
        pipeline_state = base.State(
            q=jnp.zeros(1),
            qd=jnp.zeros(1),
            x=base.Transform.create(pos=jnp.zeros(1)),
            xd=base.Motion.create(vel=jnp.zeros(1)),
            contact=None,
        )
        obs = jnp.zeros(1)
        reward, done = jnp.array(0.0), jnp.array(0)
        return ExtendedState(pipeline_state, obs, reward, done, rng)
    
    def step(self, state: ExtendedState, action: jax.Array) -> ExtendedState:
        assert state.pipeline_state is not None
        self._step_count += 1
        return state.replace(obs=jnp.array([0.0]), reward=jnp.array(1.0))
    
    @property
    def observation_size(self):
        return 1
    
    @property
    def action_size(self):
        return 1
    
    @property
    def step_count(self):
        return self._step_count
    

class SecondEnv(PipelineEnv):
    """One action, random observation in [-1, 1], one timestep long, reward = observation."""
    def __init__(
            self,
            obs_mode: ObservationMode = ObservationMode.NDARRAY,
            **kwargs,
    ):
        self._obs_mode = ObservationMode(obs_mode)
        self._step_count = 0

    def reset(self, rng: jax.Array) -> ExtendedState:
        rng, rng1 = jax.random.split(rng)
        pipeline_state = base.State(
            q=jnp.zeros(1),
            qd=jnp.zeros(1),
            x=base.Transform.create(pos=jnp.zeros(1)),
            xd=base.Motion.create(vel=jnp.zeros(1)),
            contact=None,
        )
        obs = jax.random.uniform(rng1, (1,), minval=-1.0, maxval=1.0)
        reward, done = jnp.array(0), jnp.array(0)
        return ExtendedState(pipeline_state, obs, reward, done, rng)
    
    def step(self, state: ExtendedState, action: jax.Array) -> ExtendedState:
        assert state.pipeline_state is not None
        rng, rng1 = jax.random.split(state.rng)
        obs = jax.random.uniform(rng1, (1,), minval=-1.0, maxval=1.0)
        self._step_count += 1
        return state.replace(obs=obs, reward=jnp.copy(obs), rng=rng)
    
    @property
    def observation_size(self):
        return 1
    
    @property
    def action_size(self):
        return 1
    
    @property
    def step_count(self):
        return self._step_count

class ThirdEnv(PipelineEnv):
    """One action, zero-then-one observation, two timesteps long, +1 reward at second timestep only."""
    def __init__(
            self,
            obs_mode: ObservationMode = ObservationMode.NDARRAY,
            **kwargs,
    ):
        self._obs_mode = ObservationMode(obs_mode)
        self._step_count = 0

    def reset(self, rng: jax.Array) -> ExtendedState:
        pipeline_state = base.State(
            q=jnp.zeros(1),
            qd=jnp.zeros(1),
            x=base.Transform.create(pos=jnp.zeros(1)),
            xd=base.Motion.create(vel=jnp.zeros(1)),
            contact=None,
        )
        obs = jnp.zeros(1)
        reward, done = jnp.array(0.0), jnp.array(0)
        return ExtendedState(pipeline_state, obs, reward, done, rng)
    
    def step(self, state: ExtendedState, action: jax.Array) -> ExtendedState:
        assert state.pipeline_state is not None
        self._step_count += 1
        if self._step_count == 1:
            return state.replace(obs=jnp.array(0.0))
        elif self._step_count == 2:
            return state.replace(obs=jnp.array(1.0), reward=jnp.array(1.0))
    
    @property
    def observation_size(self):
        return 1
    
    @property
    def action_size(self):
        return 1    
    
    @property
    def step_count(self):
        return self._step_count
    
class FourthEnv(PipelineEnv):
    """Two actions, zero observation, one timestep long, action-dependent +1/-1 reward."""
    def __init__(
            self,
            obs_mode: ObservationMode = ObservationMode.NDARRAY,
            **kwargs,
    ):
        self._obs_mode = ObservationMode(obs_mode)
        self._step_count = 0

    def reset(self, rng: jax.Array) -> ExtendedState:
        pipeline_state = base.State(
            q=jnp.zeros(1),
            qd=jnp.zeros(1),
            x=base.Transform.create(pos=jnp.zeros(1)),
            xd=base.Motion.create(vel=jnp.zeros(1)),
            contact=None,
        )
        obs = jnp.zeros(1)
        reward, done = jnp.array(0.0), jnp.array(0)
        return ExtendedState(pipeline_state, obs, reward, done, rng)
    
    def step(self, state: ExtendedState, action: jax.Array) -> ExtendedState:
        assert state.pipeline_state is not None
        self._step_count += 1
        reward = - jnp.abs(action[0] - 0.5)
        return state.replace(reward=reward)
        
    @property
    def observation_size(self):
        return 1
    
    @property
    def action_size(self):
        return 1
    
    @property
    def step_count(self):
        return self._step_count
    
class FifthEnv(PipelineEnv):
    """Two actions, random observation in [-1, 1], one timestep long, action and observation-dependent reward."""
    def __init__(
            self,
            obs_mode: ObservationMode = ObservationMode.NDARRAY,
            **kwargs,
    ):
        self._obs_mode = ObservationMode(obs_mode)
        self._step_count = 0

    def reset(self, rng: jax.Array) -> ExtendedState:
        rng, rng1 = jax.random.split(rng)
        pipeline_state = base.State(
            q=jnp.zeros(1),
            qd=jnp.zeros(1),
            x=base.Transform.create(pos=jnp.zeros(1)),
            xd=base.Motion.create(vel=jnp.zeros(1)),
            contact=None,
        )
        obs = jax.random.uniform(rng1, (1,), minval=-1.0, maxval=1.0)
        reward, done = jnp.copy(obs), jnp.array(0)
        return ExtendedState(pipeline_state, obs, reward, done, rng)
    
    def step(self, state: ExtendedState, action: jax.Array) -> ExtendedState:
        assert state.pipeline_state is not None
        rng, rng1 = jax.random.split(state.rng)
        obs = jax.random.uniform(rng1, (1,), minval=-1.0, maxval=1.0)
        self._step_count += 1
        reward = - jnp.abs(action[0] - state.obs[0])
        return state.replace(obs=obs, reward=reward, rng=rng)
    
    @property
    def observation_size(self):
        return 1
    
    @property
    def action_size(self):
        return 1
    
    @property
    def step_count(self):
        return self._step_count