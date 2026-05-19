import time
from typing import Optional

import gym
from gym import spaces
import numpy as np

from _settings import Settings
import control
import dqn
import sumo
import prediction


class JerkEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, config):
        super().__init__()
        sumo.start_sumo()
        self.num_cars_ahead = config.get("num_cars_ahead", Settings.CARS_AHEAD)
        self.num_cars_behind = config.get("num_cars_behind", Settings.CARS_BEHIND)
        self.max_episode_length = config.get("max_episode_length", Settings.MAX_EPISODE_LENGTH)
        self.wait_before_start = config.get("wait_before_start", 20)
        self.penalty_for_invalid_action = config.get("invalid_action_penalty", Settings.INVALID_ACTION_PENALTY)
        self.max_episode_ticks = self.max_episode_length / Settings.TICK_LENGTH
        self.current_episode_ticks = 0
        self.reward_function = dqn.get_reward_function()
        self.previous_state: Optional[prediction.HighwayState] = None
        self.previous_acceleration = None
        self.position_history = []
        self.control_history = []
        self.speed_history = []
        self.acceleration_history = []
        self.jerk_history = []
        self.state_history = []
        self.crashed = False
        self.merged = False
        self.invalid_action_reward = 0
        self.projected_jerk = 0
        self.start_time = 0
        # Observation in the form:
        # ((possibly relative) speed (float), relative_position (float), present (1 or 0)) for cars ahead then cars behind
        # Followed by (ego_speed (float), ego_acceleration (float), ego_x (float), ego_y (float))
        if Settings.USE_ACCELERATION_OF_OTHER_CARS:
            self.observation_dim = 4*(self.num_cars_ahead + self.num_cars_behind) + 4
        else:
            self.observation_dim = 3*(self.num_cars_ahead + self.num_cars_behind) + 4
        lows = np.zeros(self.observation_dim)
        highs = np.zeros(self.observation_dim)
        if Settings.NORMALIZE_VECTOR_INPUT:
            lows -= 1
            highs += 1
        else:
            for i in range(self.num_cars_ahead + self.num_cars_behind):
                if Settings.USE_ACCELERATION_OF_OTHER_CARS:
                    lows[4 * i + 0] = -9
                    lows[4 * i + 1] = -Settings.MAX_SPEED
                    lows[4 * i + 2] = -300
                    lows[4 * i + 3] = 0
                    highs[4 * i] = 6
                    highs[4 * i + 1] = Settings.MAX_SPEED + 1E-5
                    highs[4 * i + 2] = 300
                    highs[4 * i + 3] = 1
                else:
                    lows[3*i+0] = -Settings.MAX_SPEED
                    lows[3*i+1] = -300
                    lows[3*i+2] = 0
                    highs[3*i] = Settings.MAX_SPEED + 1E-5
                    highs[3*i+1] = 300
                    highs[3*i+2] = 1
            lows[-4] = 0
            highs[-4] = Settings.MAX_SPEED + 1E-5
            lows[-3] = Settings.MAX_NEGATIVE_ACCELERATION - 1E-5
            highs[-3] = Settings.MAX_POSITIVE_ACCELERATION + 1E-5
            lows[-2] = -250
            highs[-2] = 250
            lows[-1] = -10
            highs[-1] = 100
        self.observation_space = spaces.Box(low=lows, high=highs)
        self.action_space = spaces.Discrete(len(Settings.JERK_VALUES_DQN))
        self._zero_obs = np.zeros(self.observation_dim)
        self._empty_state = prediction.HighwayState.empty_state()
        self._timing_reward_s = 0.0
        self._timing_state_s = 0.0
        self._timing_vector_s = 0.0
        self._timing_steps = 0

    def _handle_jerk(self, selected_jerk):
        projected_acceleration = self.previous_acceleration + selected_jerk * Settings.TICK_LENGTH
        projected_speed = self.previous_state.ego_speed + projected_acceleration * Settings.TICK_LENGTH
        if projected_acceleration > Settings.MAX_POSITIVE_ACCELERATION or projected_acceleration < Settings.MAX_NEGATIVE_ACCELERATION:
            self.invalid_action_reward = self.penalty_for_invalid_action * Settings.TICK_LENGTH
            projected_acceleration = np.clip(projected_acceleration, Settings.MAX_NEGATIVE_ACCELERATION, Settings.MAX_POSITIVE_ACCELERATION)
        elif projected_speed > Settings.MAX_SPEED or projected_speed < 0:
            self.invalid_action_reward = self.penalty_for_invalid_action * Settings.TICK_LENGTH
            projected_speed = np.clip(projected_speed, 0, Settings.MAX_SPEED)
            projected_acceleration = (projected_speed - self.previous_state.ego_speed) / Settings.TICK_LENGTH
        else:
            self.invalid_action_reward = 0
        self.projected_jerk = (projected_acceleration - self.previous_acceleration) / Settings.TICK_LENGTH
        control.set_ego_jerk(selected_jerk)

    def _do_action(self, action):
        selected_jerk = Settings.JERK_VALUES_DQN[action]
        self._handle_jerk(selected_jerk)

    def step(self, action):
        self.control_history.append(action)
        self.current_episode_ticks += 1
        self._do_action(action)
        control.step()

        if control.just_had_collision():
            vector_state = self._zero_obs.copy()
            t0 = time.perf_counter()
            reward = self.reward_function(self._empty_state, self.projected_jerk, True, False) + self.invalid_action_reward
            self._timing_reward_s += time.perf_counter() - t0
            self._timing_steps += 1
            self.crashed = True
            return vector_state, reward, True, {}
        elif control.ego_just_arrived():
            vector_state = self._zero_obs.copy()
            self.merged = True
            t0 = time.perf_counter()
            reward = self.reward_function(self._empty_state, self.projected_jerk, False, True) + self.invalid_action_reward
            self._timing_reward_s += time.perf_counter() - t0
            self._timing_steps += 1
            return vector_state, reward, True, {}
        elif self.current_episode_ticks >= self.max_episode_ticks:
            t_state = time.perf_counter()
            state = prediction.HighwayState.from_sumo()
            self._timing_state_s += time.perf_counter() - t_state
            t_vec = time.perf_counter()
            vector_state = dqn.get_state_vector_from_base_state(state)
            self._timing_vector_s += time.perf_counter() - t_vec
            current_acceleration = state.ego_acceleration
            jerk = (current_acceleration - self.previous_acceleration) / Settings.TICK_LENGTH
            t0 = time.perf_counter()
            reward = self.reward_function(state, jerk, False, False) + self.invalid_action_reward
            self._timing_reward_s += time.perf_counter() - t0
            self._timing_steps += 1
            control.remove_ego_car()
            control.step()
            return vector_state, reward, True, {}
        else:
            t_state = time.perf_counter()
            state = prediction.HighwayState.from_sumo()
            self._timing_state_s += time.perf_counter() - t_state
            t_vec = time.perf_counter()
            vector_state = dqn.get_state_vector_from_base_state(state)
            self._timing_vector_s += time.perf_counter() - t_vec
            current_acceleration = state.ego_acceleration
            jerk = (current_acceleration - self.previous_acceleration) / Settings.TICK_LENGTH
            t0 = time.perf_counter()
            reward = self.reward_function(state, jerk, False, False) + self.invalid_action_reward
            self._timing_reward_s += time.perf_counter() - t0
            self._timing_steps += 1
            self.previous_state = state
            self.speed_history.append(state.ego_speed)
            self.position_history.append(state.ego_position)
            self.acceleration_history.append(current_acceleration)
            self.jerk_history.append(jerk)
            self.state_history.append(state)
            self.previous_acceleration = current_acceleration
            return vector_state, reward, False, {}

    def reset(self):
        self.invalid_action_reward = 0
        self.current_episode_ticks = 0
        self._timing_reward_s = 0.0
        self._timing_state_s = 0.0
        self._timing_vector_s = 0.0
        self._timing_steps = 0
        for i in range(int(self.wait_before_start / Settings.TICK_LENGTH)):
            control.step()
        start_speed = control.get_ego_start_speed()
        control.add_ego_car(start_speed)
        control.step()
        self.previous_acceleration = 0
        state = prediction.HighwayState.from_sumo()
        self.previous_state = state
        if hasattr(self.reward_function, "reset"):
            self.reward_function.reset(state)
        self.speed_history = [start_speed]
        self.control_history = []
        self.position_history = [state.ego_position]
        self.acceleration_history = [0]
        self.jerk_history = [0]
        self.state_history = [state]
        self.crashed = False
        self.merged = False
        self.start_time = time.perf_counter()
        return dqn.get_state_vector_from_base_state(state)

    def render(self, mode='human'):
        pass

    def get_stats(self):
        return {
            "crashed": self.crashed,
            "merged": self.merged,
            "state_history": self.state_history,
            "control_history": self.control_history,
            "position_history": self.position_history,
            "speed_history": self.speed_history,
            "acceleration_history": self.acceleration_history,
            "jerk_history": self.jerk_history,
            "closest_vehicle_history": [0],
            "simulation_time_taken": len(self.state_history) * Settings.TICK_LENGTH,
            "start_time": self.start_time,
            "end_time": time.perf_counter()
        }

    def close(self):
        sumo.exit_sumo()


class AccelerationEnv(JerkEnv):

    def __init__(self, config):
        super().__init__(config)
        self.action_space = spaces.Discrete(len(Settings.ACCELERATION_VALUES_DQN))

    def _do_action(self, action):
        projected_acceleration = Settings.ACCELERATION_VALUES_DQN[action]
        projected_speed = self.previous_state.ego_speed + projected_acceleration * Settings.TICK_LENGTH
        self.projected_jerk = (projected_acceleration - self.previous_acceleration) / Settings.TICK_LENGTH
        if self.projected_jerk > Settings.MAXIMUM_POSITIVE_JERK:
            self.invalid_action_reward = self.penalty_for_invalid_action * Settings.TICK_LENGTH
            self.projected_jerk = Settings.MAXIMUM_POSITIVE_JERK
            control.set_ego_jerk(Settings.MAXIMUM_POSITIVE_JERK)
        elif self.projected_jerk < Settings.MINIMUM_NEGATIVE_JERK:
            self.invalid_action_reward = self.penalty_for_invalid_action * Settings.TICK_LENGTH
            self.projected_jerk = Settings.MINIMUM_NEGATIVE_JERK
            control.set_ego_jerk(Settings.MINIMUM_NEGATIVE_JERK)
        elif projected_speed > Settings.MAX_SPEED or projected_speed < 0:
            self.invalid_action_reward = self.penalty_for_invalid_action * Settings.TICK_LENGTH
            projected_speed = np.clip(projected_speed, 0, Settings.MAX_SPEED)
            projected_acceleration = (projected_speed - self.previous_state.ego_speed) / Settings.TICK_LENGTH
            self.projected_jerk = (projected_acceleration - self.previous_acceleration) / Settings.TICK_LENGTH
            control.set_ego_speed(projected_speed)
        else:
            self.invalid_action_reward = 0
            control.set_ego_speed(projected_speed)


class ContinuousJerkEnv(JerkEnv):

    def __init__(self, config):
        super().__init__(config)
        self.action_space = spaces.Box(
            Settings.MINIMUM_NEGATIVE_JERK, Settings.MAXIMUM_POSITIVE_JERK, shape=(1,)
        )

    def _do_action(self, action):
        self._handle_jerk(action)


class _GymnasiumAdapter:
    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
        return obs, reward, terminated, truncated, info

    def close(self):
        return self._env.close()


def _make_gymnasium_jerk_env(**kwargs):
    config = kwargs.get("config", {})
    return _GymnasiumAdapter(JerkEnv(config))


def _make_gymnasium_accel_env(**kwargs):
    config = kwargs.get("config", {})
    return _GymnasiumAdapter(AccelerationEnv(config))


def _make_gymnasium_continuous_jerk_env(**kwargs):
    config = kwargs.get("config", {})
    return _GymnasiumAdapter(ContinuousJerkEnv(config))


def register_environments():
    gym.envs.register(
        id='sumo-jerk-v0',
        entry_point='merge_gym:JerkEnv',
        max_episode_steps=500,
        kwargs={'config': {}},
    )
    gym.envs.register(
        id='sumo-accel-v0',
        entry_point='merge_gym:AccelerationEnv',
        max_episode_steps=500,
        kwargs={'config': {}},
    )
    gym.envs.register(
        id='sumo-jerk-continuous-v0',
        entry_point='merge_gym:ContinuousJerkEnv',
        max_episode_steps=500,
        kwargs={'config': {}},
    )

    try:
        from gymnasium.envs.registration import register as gymnasium_register

        gymnasium_register(
            id='sumo-jerk-v0',
            entry_point=_make_gymnasium_jerk_env,
            max_episode_steps=500,
            kwargs={'config': {}},
            disable_env_checker=True,
        )
        gymnasium_register(
            id='sumo-accel-v0',
            entry_point=_make_gymnasium_accel_env,
            max_episode_steps=500,
            kwargs={'config': {}},
            disable_env_checker=True,
        )
        gymnasium_register(
            id='sumo-jerk-continuous-v0',
            entry_point=_make_gymnasium_continuous_jerk_env,
            max_episode_steps=500,
            kwargs={'config': {}},
            disable_env_checker=True,
        )
    except Exception:
        pass


class GymEnvironment:
    def __init__(self, env_name: str, device: str = "cpu", **make_kwargs):
        register_environments()
        self.name = env_name
        self.device = device

        gymnasium = None
        try:
            import gymnasium as gymnasium  # type: ignore
        except Exception:
            gymnasium = None

        if gymnasium is not None:
            try:
                self._env = gymnasium.make(env_name, **make_kwargs)
                return
            except Exception:
                pass

        self._env = gym.make(env_name, **make_kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()


if not hasattr(_GymnasiumAdapter, 'unwrapped'):
    _GymnasiumAdapter.unwrapped = property(lambda self: self)
if not hasattr(_GymnasiumAdapter, '__getattr__'):
    def _adapter_getattr(self, name):
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if not hasattr(self._env, name):
            raise AttributeError(f"Wrapped env '{self._env.__class__.__name__}' has no attribute '{name}'")
        return getattr(self._env, name)
    _GymnasiumAdapter.__getattr__ = _adapter_getattr
