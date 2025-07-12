from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, MutableMapping, Tuple

@dataclass
class StepBuffers:
    n_obs: int
    obs: Deque[Any] = field(init=False)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: Dict[str, Deque[Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.obs = deque(maxlen=self.n_obs + 1)
        self.infos = {}

    def add_obs(self, o : Any) -> None:
        """Append a single observation frame."""
        self.obs.append(o)

    def add_reward(self, r : float | int) -> None:
        """Append a scalar reward (cast to float)."""
        self.rewards.append(float(r))

    def add_done(self, d : bool) -> None:
        """Append a boolean done flag."""
        self.dones.append(bool(d))

    def add_info(self, info : MutableMapping[str, Any]) -> None:
        """Append the info dict returned by the environment."""
        for k, v in info.items():
            self.infos.setdefault(k, deque(maxlen=self.n_obs + 1)).append(v)

    def reduce(self, agg : str ="max") -> Tuple[float, bool]:
        """Aggregate inner-step rewards and done flags for the current step.

        Parameters
        ----------
        agg : str
            Reduction function to apply to the reward list.

        Returns
        -------
        Tuple[float, bool]
            `(reward, done)` where `reward` is the aggregated scalar and `done`
            is `any(self.dones)`.
        """
        import numpy as np
        agg_fn_map: Dict[str, Any] = {
                "max": np.max,
                "min": np.min,
                "mean": np.mean,
                "sum": np.sum,
        }
        if agg not in agg_fn_map:
            raise ValueError(f"Unsupported aggregation `{agg}`")
        reward: float = float(agg_fn_map[agg](self.rewards))
        done: bool = bool(np.max(self.dones))
        return reward, done
