"""Interaction datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mairl.core.math import validate_policy
from mairl.core.problem import FloatArray
from mairl.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class InteractionDataset:
    policies: FloatArray
    follower_actions: np.ndarray

    def __post_init__(self) -> None:
        if self.policies.ndim != 2:
            raise ConfigurationError("dataset policies must be two-dimensional")
        if self.follower_actions.shape != (self.policies.shape[0],):
            raise ConfigurationError("dataset policies and actions have inconsistent lengths")

    @property
    def size(self) -> int:
        return int(self.follower_actions.size)

    @classmethod
    def empty(cls, n_leader_actions: int) -> InteractionDataset:
        return cls(
            policies=np.empty((0, n_leader_actions), dtype=np.float64), follower_actions=np.empty(0, dtype=np.int64)
        )

    def append(self, policy: FloatArray, follower_action: int) -> InteractionDataset:
        mu = validate_policy(policy, self.policies.shape[1])
        return InteractionDataset(
            policies=np.concatenate((self.policies, mu[None, :]), axis=0),
            follower_actions=np.concatenate((self.follower_actions, np.asarray([follower_action], dtype=np.int64))),
        )

    @classmethod
    def from_rows(cls, policies: list[FloatArray], actions: list[int], n_leader_actions: int) -> InteractionDataset:
        if not policies:
            return cls.empty(n_leader_actions)
        matrix = np.stack([validate_policy(policy, n_leader_actions) for policy in policies], axis=0)
        return cls(matrix, np.asarray(actions, dtype=np.int64))
