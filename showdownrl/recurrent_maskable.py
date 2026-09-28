"""
Recurrent Maskable PPO — combines RecurrentPPO's GRU/LSTM with MaskablePPO's action masking.

Provides:
  - RecurrentMaskableRolloutBufferSamples
  - RecurrentMaskableRolloutBuffer
  - RecurrentMaskableActorCriticPolicy
  - RecurrentMaskablePPO

Usage in train_ppo.py:
    algorithm_class, _ = resolve_algorithm("recurrent_maskable_ppo")
    model = algorithm_class("RecurrentMaskableActorCriticPolicy", env, ...)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Generator, NamedTuple

import gymnasium as sp
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.maskable.distributions import (
    MaskableDistribution,
    make_masked_proba_distribution,
)
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.recurrent.buffers import (
    RecurrentRolloutBuffer,
    create_sequencers,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

# ─── Modified namedtuple ───────────────────────────────────────────────────

_RecurrentMaskableSamplesBase = NamedTuple(
    "RecurrentMaskableRolloutBufferSamples",
    [
        ("observations", th.Tensor),
        ("actions", th.Tensor),
        ("old_values", th.Tensor),
        ("old_log_prob", th.Tensor),
        ("advantages", th.Tensor),
        ("returns", th.Tensor),
        ("lstm_states", RNNStates),
        ("episode_starts", th.Tensor),
        ("mask", th.Tensor),
        ("action_masks", th.Tensor),
    ],
)


class RecurrentMaskableRolloutBufferSamples(_RecurrentMaskableSamplesBase):
    """Same as RecurrentRolloutBufferSamples but with action_masks."""


# ─── Recurrent Maskable Rollout Buffer ─────────────────────────────────────

class RecurrentMaskableRolloutBuffer(RecurrentRolloutBuffer):
    """Extends RecurrentRolloutBuffer to store action masks alongside LSTM states."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: sp.Space,
        action_space: sp.Space,
        hidden_state_shape: tuple[int, int, int, int],
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.action_masks: np.ndarray | None = None
        self.mask_dims: int = 0
        super().__init__(
            buffer_size, observation_space, action_space,
            hidden_state_shape, device, gae_lambda, gamma, n_envs,
        )

    def reset(self) -> None:
        super().reset()
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("RecurrentMaskablePPO only supports Discrete action spaces")
        self.mask_dims = int(self.action_space.n)
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32
        )

    def add(self, *args, lstm_states: RNNStates, action_masks: np.ndarray | None = None, **kwargs) -> None:
        if action_masks is not None and self.action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def get(self, batch_size: int | None = None) -> Generator[RecurrentMaskableRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        if not self.generator_ready:
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            fields_to_flatten = [
                "observations", "actions", "values", "log_probs",
                "advantages", "returns",
                "hidden_states_pi", "cell_states_pi",
                "hidden_states_vf", "cell_states_vf",
                "episode_starts",
            ]
            if self.action_masks is not None:
                fields_to_flatten.append("action_masks")

            for tensor in fields_to_flatten:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx: start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: VecNormalize | None = None,
    ) -> RecurrentMaskableRolloutBufferSamples:
        seq_start_indices, local_pad, local_pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )
        n_seq = len(seq_start_indices)
        max_length = local_pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length

        lstm_states_pi = (
            self.hidden_states_pi[batch_inds][seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            self.hidden_states_vf[batch_inds][seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (
            self.to_torch(lstm_states_pi[0]).contiguous(),
            self.to_torch(lstm_states_pi[1]).contiguous(),
        )
        lstm_states_vf = (
            self.to_torch(lstm_states_vf[0]).contiguous(),
            self.to_torch(lstm_states_vf[1]).contiguous(),
        )

        # Build action_masks with padding
        assert self.action_masks is not None
        action_masks_padded = local_pad(
            self.action_masks[batch_inds], padding_value=1.0
        ).reshape((padded_batch_size, self.mask_dims))

        return RecurrentMaskableRolloutBufferSamples(
            observations=local_pad(self.observations[batch_inds]).reshape(
                (padded_batch_size, *self.obs_shape)
            ),
            actions=local_pad(self.actions[batch_inds]).reshape(
                (padded_batch_size, *self.actions.shape[1:])
            ),
            old_values=local_pad_and_flatten(self.values[batch_inds]),
            old_log_prob=local_pad_and_flatten(self.log_probs[batch_inds]),
            advantages=local_pad_and_flatten(self.advantages[batch_inds]),
            returns=local_pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=local_pad_and_flatten(self.episode_starts[batch_inds]),
            mask=local_pad_and_flatten(np.ones_like(self.returns[batch_inds])),
            action_masks=action_masks_padded,
        )


# ─── Recurrent Maskable Actor-Critic Policy ────────────────────────────────

class RecurrentMaskableActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    Recurrent actor-critic policy that also supports action masking.

    Combines RecurrentActorCriticPolicy's LSTM with MaskablePolicy's
    action masking in forward(), evaluate_actions(), _predict(), and get_distribution().
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("RecurrentMaskablePPO only supports Discrete action spaces")
        # The parent builds action_net with the standard distribution. Maskable
        # distributions use the same logits shape, so only the distribution
        # object itself needs replacing after parent initialization.
        self.action_dist = make_masked_proba_distribution(self.action_space)

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> tuple[MaskableDistribution, tuple[th.Tensor, ...]]:
        distribution, lstm_states = super().get_distribution(
            obs, lstm_states, episode_starts
        )
        assert isinstance(distribution, MaskableDistribution)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution, lstm_states

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """Build the maskable action distribution from actor logits."""
        action_logits = self.action_net(latent_pi)
        assert isinstance(self.action_dist, MaskableDistribution)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
        distribution, lstm_states = self.get_distribution(
            observation, lstm_states, episode_starts, action_masks
        )
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Predict with both recurrence AND action masking."""
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]

        if state is None:
            state = np.concatenate(
                [np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1
            )
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            states = (
                th.tensor(state[0], dtype=th.float32, device=self.device),
                th.tensor(state[1], dtype=th.float32, device=self.device),
            )
            episode_starts = th.tensor(
                episode_start, dtype=th.float32, device=self.device
            )
            actions, states = self._predict(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states


# ─── Recurrent Maskable PPO Algorithm ──────────────────────────────────────

class RecurrentMaskablePPO(RecurrentPPO):
    """
    Recurrent PPO with invalid action masking support.

    Extends RecurrentPPO by:
      - Overriding _setup_model() to use RecurrentMaskableRolloutBuffer
      - Overriding collect_rollouts() to fetch action masks and pass them
      - Overriding train() to pass action_masks during evaluate_actions
      - Using RecurrentMaskableActorCriticPolicy
    """

    # Register the maskable policy so "MlpLstmPolicy" string works
    policy_aliases = {
        "MlpLstmPolicy": RecurrentMaskableActorCriticPolicy,
    }

    def _setup_model(self) -> None:
        # Let RecurrentPPO initialize the policy, recurrent states, and schedules,
        # then replace only its rollout buffer with the mask-aware variant.
        super()._setup_model()
        if not isinstance(self.policy, RecurrentMaskableActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (
            self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size
        )
        self.rollout_buffer = RecurrentMaskableRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Run recurrent inference while applying the current action masks."""
        return self.policy.predict(
            observation,
            state,
            episode_start,
            deterministic,
            action_masks=action_masks,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert isinstance(rollout_buffer, RecurrentMaskableRolloutBuffer), (
            f"{rollout_buffer} doesn't support recurrent + maskable policy"
        )
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(
                    self._last_episode_starts, dtype=th.float32, device=self.device
                )
                # Get action masks from env
                action_masks = get_action_masks(env)

                actions, values, log_probs, lstm_states = self.policy(
                    obs_tensor, lstm_states, episode_starts,
                    action_masks=action_masks,
                )

            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx: idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx: idx + 1, :].contiguous(),
                        )
                        episode_starts_t = th.tensor(
                            [False], dtype=th.float32, device=self.device
                        )
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state, episode_starts_t
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                action_masks=action_masks,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (
                        advantages[mask].std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()[mask]
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = th.mean(
                    ((rollout_data.returns - values_pred) ** 2)[mask]
                )
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        ((th.exp(log_ratio) - 1) - log_ratio)[mask]
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: "
                            f"{approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item()
            )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
