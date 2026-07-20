"""Minimal stand-in for ``stable_baselines3.common.distributions``.

PCLA port: MindDrive's detector builds a ``CategoricalDistribution(action_dim=7)``
in ``__init__`` to sample the discrete meta-action (speed command), so it is on the
closed-loop inference path and cannot simply be dropped.

Upstream imports it from stable-baselines3, whose package import pulls in
``gymnasium`` and whose install would risk upgrading ``pandas``/``matplotlib`` in
the shared PCLA environment that ~20 other agents rely on. The class itself is a
thin wrapper over ``torch.distributions.Categorical``, so it is reproduced here
with matching semantics and no third-party dependency.

Faithful to stable-baselines3 (BSD-3-Clause), see
https://github.com/DLR-RM/stable-baselines3 -- ``common/distributions.py``.
"""

import torch as th
from torch import nn
from torch.distributions import Categorical

__all__ = ['CategoricalDistribution']


class CategoricalDistribution:
    """Categorical distribution over discrete actions.

    :param action_dim: number of discrete actions.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Layer producing the logits of the categorical distribution."""
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, action_logits: th.Tensor) -> 'CategoricalDistribution':
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, action_logits: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor):
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)
