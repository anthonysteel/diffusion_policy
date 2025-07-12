from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
from .space_utils import repeated_space, stack_last
from .buffers import StepBuffers
from typing import Sequence, Tuple, Mapping
import numpy as np

class MultiStep(gym.Wrapper):
    """Gym wrapper for multi-step action repeat & frame-stacking.
    """

    def __init__(self,
            env : gym.Env,
            n_obs : int = 4,
            n_action : int = 2,
            reward_reduce : str = 'max',
            max_episode_steps : int | None = None
    ) -> None:
        """__init__.

        Parameters
        ----------
        env : gym.Env
            env
        n_obs : int
            n_obs
        n_action : int
            n_action
        reward_reduce : str
            reward_reduce
        max_episode_steps : int | None
            max_episode_steps

        Returns
        -------
        None

        """
        super().__init__(env)
        self.n_obs = n_obs
        self.n_action = n_action
        self.reward_reduce = reward_reduce
        self.max_ep = max_episode_steps

        self._obs_space = repeated_space(env.observation_space, n_obs)
        self._act_space = repeated_space(env.action_space, n_action)
        self.buf = StepBuffers(n_obs)

    @property
    def observation_space(self) -> spaces.Spaces:
        """observation_space.

        Parameters
        ----------

        Returns
        -------
        spaces.Spaces

        """
        return self._obs_space

    @property
    def action_space(self) -> spaces.Space:
        """action_space.

        Parameters
        ----------

        Returns
        -------
        spaces.Space

        """
        return self_act_space

    def reset(self, **kw) -> np.ndarray:
        """Reset the underlying environment and clear all buffers.

        Parameters
        ----------
        kw :
            kw

        Returns
        -------
        np.ndarray

        """
        o = self.env.reset(**kw)
        self.buf = StepBuffers(self.n_obs)
        self.buf.add_obs(o)
        return self._stack_obs()

    def step(self,
            action: Sequence[np.ndarray]
   ) -> Tuple[np.ndarray, float, bool, Mapping[str, np.ndarray]]:
        """Execute `n_action` inner steps and aggregate results.

        Parameters
        ----------
        action : Sequence[np.ndarray]
            Sequence of length `n_action` whose items are valid inputs for the
            wrapped environment's `step`.

        Returns
        -------
        Tuple[np.ndarray, float, bool, Mapping[str, np.ndarray]]
        
        stacked_obs : np.ndarray
            New observation stack.

        reward : float
            Aggregated scalar reward for the macro-step

        done : bool
            `True` if any inner step (or the episode-length gaurd) signals
            termination.

        info : Mapping[str, np.ndarray]
            Dict whose values are stacked to match the observation hitory.
        """
        assert len(action) == self.n_action
        for sub_act in action:
            if self.buf.dones and self.buf.dones[-1]:
                break
            obs, rew, terminated, truncated, info = self.env.step(sub_act)
            done = terminated or truncated
            self.buf.add_obs(obs)
            self.buf.add_reward(rew)
            self.buf.add_done(done)
            self.buf.add_info(info)

        r_red, d_red = self.buf.reduce(self.reward_reduce)
        if self.max_ep and len(self.buf.rewards) >= self.max_ep:
            d_red = True
        return self._stack_obs(), r_red, d_red, {
                k : stack_last(v, self.n_obs) for k, v in self.buf.infos.items()
        }

    def _stack_obs(self) -> np.ndarray:
        """Return the most recent `n_obs` frames as one stacked array."""
        head = self.buf.obs[-self.n_obs:]
        return stack_last(list(head), self.n_obs)

