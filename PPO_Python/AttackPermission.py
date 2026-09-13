"""Opt-in PPO_6 navigation with episode-level attack permission."""
import numpy as np
import torch
import os
import json
from pathlib import Path
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class PermissionPolicy(MultiInputActorCriticPolicy):
    def distribution_for(self, obs, latent):
        logits = self.action_net(latent)
        # Last scalar: 0 = unrestricted, 1 = attacks forbidden.
        forbidden = obs["general"][:, -1] >= 0.5
        logits = logits.clone()
        logits[:, -1] = logits[:, -1].masked_fill(forbidden, -torch.inf)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        dist = self.distribution_for(obs, pi)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self.value_net(vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        dist = self.distribution_for(obs, pi)
        return self.value_net(vf), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        pi = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self.distribution_for(obs, pi)


class TerminalExplorationPolicy(MultiInputActorCriticPolicy):
    """Temper only prohibited-phase attack logits, consistently for PPO."""

    attack_temperature = 0.1

    def distribution_for(self, obs, latent):
        logits = self.action_net(latent)
        costly = (obs["general"][:, -1] >= 0.5).unsqueeze(1)
        attack_logits = torch.where(
            costly,
            logits[:, -2:] * float(self.attack_temperature),
            logits[:, -2:],
        )
        logits = torch.cat([logits[:, :-2], attack_logits], dim=1)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        dist = self.distribution_for(obs, pi)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self.value_net(vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        dist = self.distribution_for(obs, pi)
        return self.value_net(vf), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        pi = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self.distribution_for(obs, pi)


EXPLORATION_SOURCE_STEP = 51_843_364
EXPLORATION_ANNEAL_STEPS = 500_000
EXPLORATION_INITIAL_TEMPERATURE = 0.1
CURRICULUM_DURATIONS = (1, 2, 4, 8, 16, 32, 64, 100, 150)
CURRICULUM_TEMPERATURES = (0.1, 0.25, 0.5, 1.0)
CURRICULUM_WINDOW = 100
CURRICULUM_PROMOTION_RATE = 0.8
CURRICULUM_DEMOTION_RATE = 0.4


def exploration_temperature(global_step):
    progress = np.clip(
        (int(global_step) - EXPLORATION_SOURCE_STEP)
        / EXPLORATION_ANNEAL_STEPS,
        0.0,
        1.0,
    )
    return float(
        EXPLORATION_INITIAL_TEMPERATURE
        + (1.0 - EXPLORATION_INITIAL_TEMPERATURE) * progress
    )


def configure_exploration(model, temperature=None):
    if isinstance(model.policy, TerminalExplorationPolicy):
        model.policy.attack_temperature = (
            exploration_temperature(model.num_timesteps)
            if temperature is None
            else float(temperature)
        )


def new_curriculum_state():
    return {
        "version": 1,
        "duration_index": 0,
        "temperature_index": 0,
        "recent_outcomes": [],
        "total_prohibited_phases": 0,
        "promotions": 0,
        "demotions": 0,
    }


def curriculum_values(state):
    return (
        CURRICULUM_DURATIONS[int(state["duration_index"])],
        CURRICULUM_TEMPERATURES[int(state["temperature_index"])],
    )


def record_curriculum_outcome(state, survived):
    state = dict(state)
    outcomes = list(state["recent_outcomes"])
    outcomes.append(int(bool(survived)))
    outcomes = outcomes[-CURRICULUM_WINDOW:]
    state["recent_outcomes"] = outcomes
    state["total_prohibited_phases"] += 1
    event = None
    if len(outcomes) == CURRICULUM_WINDOW:
        rate = sum(outcomes) / len(outcomes)
        duration_index = int(state["duration_index"])
        temperature_index = int(state["temperature_index"])
        if rate >= CURRICULUM_PROMOTION_RATE:
            if temperature_index < len(CURRICULUM_TEMPERATURES) - 1:
                state["temperature_index"] = temperature_index + 1
                event = "temperature_promoted"
            elif duration_index < len(CURRICULUM_DURATIONS) - 1:
                state["duration_index"] = duration_index + 1
                state["temperature_index"] = 0
                event = "duration_promoted"
            if event is not None:
                state["promotions"] += 1
                state["recent_outcomes"] = []
        elif rate < CURRICULUM_DEMOTION_RATE and (
            duration_index > 0 or temperature_index > 0
        ):
            if temperature_index > 0:
                state["temperature_index"] = temperature_index - 1
            else:
                state["duration_index"] = duration_index - 1
                state["temperature_index"] = len(
                    CURRICULUM_TEMPERATURES
                ) - 1
            state["demotions"] += 1
            state["recent_outcomes"] = []
            event = "demoted"
    return state, event


def load_curriculum(path):
    path = Path(path)
    if not path.is_file():
        return new_curriculum_state()
    state = json.loads(path.read_text(encoding="utf-8"))
    curriculum_values(state)
    return state


def save_curriculum(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def condition(processed, forbidden):
    return {**processed, "general": np.concatenate([
        processed["general"], np.asarray([forbidden], dtype=np.float32)
    ])}


def execute_action(action, forbidden):
    result = np.array(action, copy=True)
    if forbidden:
        result[6] = 0
    return result


FORBIDDEN_NO_DEPTH_STEPS = 150  # 30 simulated seconds at 5 Hz.


def stalled_attempt(forbidden, steps_since_best_depth):
    return bool(forbidden and steps_since_best_depth >= FORBIDDEN_NO_DEPTH_STEPS)


CONTRACT = {
    "phase": "PPO_6", "experiment": "binary_attack_permission",
    "source_step": 51843364, "model_schema_version": 2,
    "permission": "last general scalar: 0 allowed, 1 forbidden",
    "forbidden_probability": 0.5, "resample": "episode reset",
    "mask": "attack-on logit disabled in sampling and PPO evaluation",
    "execution_guard": True,
    "forbidden_watchdog_penalty": 0.0,
    "forbidden_no_new_best_depth_steps": FORBIDDEN_NO_DEPTH_STEPS,
    "allowed_watchdog_penalty": -10.0,
    "progress_and_completion_reward_unchanged": True,
    "miss_penalty_enabled": False,
}


def contract():
    if os.environ.get("PPO_ATTACK_TERMINAL", "0") == "1":
        result = {
            "phase": "PPO_6",
            "experiment": "alternating_terminal_attack_violation",
            "source_step": 51843364,
            "model_schema_version": 2,
            "attack_cost": "last general scalar: 0 allowed, 1 prohibited",
            "phase_steps": 150,
            "initial_phase": "allowed",
            "resample": "alternate every 150 decisions",
            "attack_available_in_policy": True,
            "attack_execution_violation": "terminate episode",
            "violation_signal": "actual swing accepted by the game",
            "violation_penalty": -10.0,
            "prohibited_no_new_best_depth_steps": 150,
            "other_watchdog_penalty": -10.0,
            "termination_before_switch": True,
            "per_swing_penalty": 0.0,
            "miss_penalty_enabled": False,
        }
        if os.environ.get("PPO_ATTACK_EXPLORATION", "0") == "1":
            result.update(
                experiment="alternating_terminal_attack_exploration",
                prohibited_attack_logit_temperature_initial=(
                    EXPLORATION_INITIAL_TEMPERATURE
                ),
                prohibited_attack_logit_temperature_final=1.0,
                temperature_anneal_steps=EXPLORATION_ANNEAL_STEPS,
                temperature_scope="attack logits during prohibited phases only",
                ppo_sampling_and_evaluation_distribution_identical=True,
            )
        if os.environ.get("PPO_ATTACK_CURRICULUM", "0") == "1":
            result.update(
                experiment="adaptive_terminal_attack_curriculum",
                prohibited_phase_durations=list(CURRICULUM_DURATIONS),
                prohibited_attack_temperatures=list(CURRICULUM_TEMPERATURES),
                stage_window=CURRICULUM_WINDOW,
                promotion_survival_rate=CURRICULUM_PROMOTION_RATE,
                demotion_survival_rate=CURRICULUM_DEMOTION_RATE,
                initial_prohibited_duration=CURRICULUM_DURATIONS[0],
                persistent_state="attack_curriculum_state.json",
                allowed_phase_steps=150,
                short_phase_depth_requirement=False,
            )
        return result
    if os.environ.get("PPO_ATTACK_PENALTY", "0") == "1":
        return {
            "phase": "PPO_6",
            "experiment": "alternating_attack_execution_penalty",
            "source_step": 51843364,
            "model_schema_version": 2,
            "attack_cost": "last general scalar: 0 free, 1 costly",
            "phase_steps": 150,
            "initial_phase": "free",
            "resample": "alternate every 150 decisions",
            "attack_available_in_all_phases": True,
            "attack_execution_penalty_scale": 0.001,
            "penalty_signal": "actual swings accepted by the game",
            "costly_no_new_best_depth_steps": 150,
            "watchdog_penalty": -10.0,
            "termination_before_switch": True,
            "miss_penalty_enabled": False,
        }
    result = dict(CONTRACT)
    if os.environ.get("PPO_ATTACK_PHASES", "0") == "1":
        result.update(experiment="alternating_attack_permission",
                      forbidden_watchdog_penalty=-10.0,
                      phase_steps=150, initial_phase="allowed",
                      resample="alternate every 150 decisions",
                      forbidden_probability=None,
                      forbidden_timer="since phase start or latest new best tile depth",
                      termination_before_switch=True)
    return result
