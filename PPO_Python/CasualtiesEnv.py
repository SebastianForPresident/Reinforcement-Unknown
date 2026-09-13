import gymnasium as gym
import numpy as np
import operator
import threading
import Reward
from EpisodeTrace import EpisodeTraceWriter
import ObservationNormalization as ObsNorm
import ProtocolValidation
import Types
import os
import AttackPermission
import ResourceBudget
from pathlib import Path

_server = None
_general_input_dim = None
_observation_lock = None
MAX_RESET_ATTEMPTS = 3
MIN_WORLD_SEED = 0
MAX_WORLD_SEED = np.iinfo(np.int32).max
MAX_EPISODE_STEPS = 100_000
MAX_STAGNANT_STEPS = 75_000
MAX_INACTIVE_STEPS = 75
MAX_MOBILE_STAGNANT_STEPS = 10_000
INACTIVE_VELOCITY_EPSILON = 0.05

# CB1 adds previous-action and absolute-position context and uses a
# semantic encoder instead of flattening raw wire values.
GENERAL_FIELD_NAMES = ObsNorm.GENERAL_FIELD_NAMES

def GetGeneralValues(obs):
    return ObsNorm.encode_general(obs)

def Init_General():
    global _general_input_dim
    dummy = np.zeros((), dtype=Types.OBSERVATION_DTYPE)
    dummy["WorldDimensions"]["X"] = 1
    dummy["WorldDimensions"]["Y"] = 1
    _general_input_dim = GetGeneralValues(dummy).size

def PreprocessObservation(obs):
    return {
        "general": GetGeneralValues(obs),
        "spatial": ObsNorm.encode_spatial(obs),
    }


def TruncationReason(
    episode_steps,
    last_best_depth_step,
    inactive_steps=0,
    max_episode_steps=MAX_EPISODE_STEPS,
    max_stagnant_steps=MAX_STAGNANT_STEPS,
    max_inactive_steps=MAX_INACTIVE_STEPS,
    mobile_stagnant_steps=0,
    max_mobile_stagnant_steps=MAX_MOBILE_STAGNANT_STEPS,
):
    if episode_steps >= max_episode_steps:
        return "max_episode_steps"
    if episode_steps - last_best_depth_step >= max_stagnant_steps:
        return "stagnant_best_depth"
    if inactive_steps >= max_inactive_steps:
        return "capable_inactivity"
    if mobile_stagnant_steps >= max_mobile_stagnant_steps:
        return "mobile_stagnation"
    return None


def AdvanceMobileStagnation(previous_steps, progress_delta):
    """Reset only for genuine new continuous depth progress."""
    if float(progress_delta) > 0.0:
        return 0
    return int(previous_steps) + 1


def InactivityStatus(obs, previous_tile_position, previous_block_health):
    """Return capability context and the physical inactivity gates."""
    capable = (
        not bool(obs["PlayerDead"])
        and float(obs["Consciousness"]) > 0.0
        and float(obs["TimeRagdolled"]) <= 0.0
    )
    tile_position = (
        int(obs["PlayerTilePosition"]["X"]),
        int(obs["PlayerTilePosition"]["Y"]),
    )
    velocity_inactive = (
        abs(float(obs["Velocity"]["X"])) <= INACTIVE_VELOCITY_EPSILON
        and abs(float(obs["Velocity"]["Y"])) <= INACTIVE_VELOCITY_EPSILON
    )
    block_health_unchanged = np.array_equal(
        obs["RelativeBlockMap"]["Health"],
        previous_block_health,
    )
    return {
        "capable": bool(capable),
        "tile_unchanged": bool(tile_position == previous_tile_position),
        "velocity_inactive": bool(velocity_inactive),
        "block_health_unchanged": bool(block_health_unchanged),
    }


def PhysicallyInactive(status):
    """Return inactivity without exempting a self-induced incapacity state."""
    return (
        bool(status["tile_unchanged"])
        and bool(status["velocity_inactive"])
        and bool(status["block_health_unchanged"])
    )


def AdvanceInactivity(
    previous_steps, status, depth_improved, count_incapacity=False
):
    """Count capable stillness or consecutive self-induced incapacity."""
    if count_incapacity and not bool(status["capable"]):
        return int(previous_steps) + 1
    if (
        bool(status["capable"])
        and not bool(depth_improved)
        and PhysicallyInactive(status)
    ):
        return int(previous_steps) + 1
    return 0

def Start(server):
    global _server, _observation_lock
    _server = server
    _observation_lock = server.observation_lock

def Decode(action):
    """Decode the seven-action CB1 policy into the sequenced wire protocol.

    The Unity harness and Server retain the 28 control fields plus a CB1
    decision sequence. Only seven controls affect locomotion and direct item
    use; every other field is explicitly reset on each policy decision.
    """
    if len(action) != 7:
        raise ValueError(f"CB1 expects 7 actions, received {len(action)}")

    _server.move = action[0] - 1
    _server.jump = action[1]
    _server.vertMove = action[2] - 1
    _server.crouch = action[3]

    _server.lookdX = action[4] - 4
    _server.lookdY = action[5] - 5

    _server.attack = action[6]

    # Legacy control fields intentionally disabled for CB1.
    _server.interact = 0
    _server.targetSlotIndex = 0
    _server.selectedSlotIndex = -1
    _server.dropItem = 0
    _server.moveItem = 0
    _server.selectedBagIndex = -1
    _server.useItem = 0
    _server.useItemWorld = 0
    _server.selectedLimb = 0
    _server.useItemMedical = 0
    _server.selectedRecipe = -1
    _server.favoriteItem = 0
    _server.switchMainHand = 0
    _server.trySleep = 0
    _server.ragdoll = 0
    _server.exercise = -1
    _server.bark = 0
    _server.throw = 0
    _server.liquidAmount = 0
    _server.drainLiquid = 0
    _server.pullLiquidFromWorld = 0

    # The bridge latches only a strictly newer sequence at a policy boundary.
    _server.action_sequence += 1

    # PPO optimizer updates leave Unity paused.  Resume only after this fresh
    # policy action has been decoded, so no physics tick can replay the action
    # that preceded the update.
    _server.SendDecodedAction()

def NormalizeWorldSeed(world_seed):
    if isinstance(world_seed, (bool, np.bool_)):
        raise ValueError("World seed must be an integer, not a boolean")

    try:
        normalized = operator.index(world_seed)
    except TypeError as exc:
        raise ValueError(f"Invalid world seed: {world_seed!r}") from exc

    if not MIN_WORLD_SEED <= normalized <= MAX_WORLD_SEED:
        raise ValueError(
            f"World seed must be in [{MIN_WORLD_SEED}, {MAX_WORLD_SEED}], "
            f"received {normalized}"
        )
    return normalized


def SendReset(reset_token, world_seed):
    if _server is None:
        raise RuntimeError("PPO server has not been started")

    world_seed = NormalizeWorldSeed(world_seed)
    _server.reset_requested.set()
    with _server.action_write_lock:
        _server.action_pipe.sendall(
            f"RESET {reset_token} {world_seed}\n".encode("ascii")
        )

class Env(gym.Env):
    def __init__(self):
        Init_General()
        self.missed_attack_penalty_mode = (
            os.environ.get("PPO_MISSED_ATTACK_PENALTY", "0") == "1"
        )
        self.attack_prior_mode = (
            os.environ.get("PPO_ATTACK_LOGIT_PRIOR", "0") == "1"
        )
        self.attack_budget_mode = (
            os.environ.get("PPO_ATTACK_BUDGET", "0") == "1"
        )
        self.natural_stamina_mode = (
            os.environ.get("PPO_NATURAL_STAMINA", "0") == "1"
        )
        self.terminal_attack_mode = os.environ.get("PPO_ATTACK_TERMINAL", "0") == "1"
        self.attack_penalty_mode = (
            os.environ.get("PPO_ATTACK_PENALTY", "0") == "1"
            or self.terminal_attack_mode
        )
        hard_permission_mode = os.environ.get("PPO_ATTACK_PERMISSION", "0") == "1"
        if self.attack_penalty_mode and hard_permission_mode:
            raise ValueError("Select attack masking or attack penalty, not both")
        self.permission_mode = hard_permission_mode or self.attack_penalty_mode
        if self.attack_budget_mode and self.permission_mode:
            raise ValueError(
                "Attack budget and legacy permission experiments are exclusive"
            )
        self.attack_budget = 0
        self.attack_budget_used = 0
        self.attack_forbidden = False
        self.attack_cost = 0.0
        self.alternating_permission = (
            os.environ.get("PPO_ATTACK_PHASES", "0") == "1"
            or self.attack_penalty_mode
        )
        self.attack_curriculum_mode = (
            os.environ.get("PPO_ATTACK_CURRICULUM", "0") == "1"
        )
        if self.attack_curriculum_mode and not (
            self.terminal_attack_mode
            and os.environ.get("PPO_ATTACK_EXPLORATION", "0") == "1"
        ):
            raise ValueError(
                "PPO_ATTACK_CURRICULUM requires terminal mode and exploration"
            )
        self.attack_curriculum_state = AttackPermission.new_curriculum_state()
        self.attack_curriculum_state_path = None
        (
            self.attack_curriculum_duration,
            self.attack_exploration_temperature,
        ) = AttackPermission.curriculum_values(self.attack_curriculum_state)
        self.attack_exploration_temperature_applied = (
            self.attack_exploration_temperature
        )
        if self.alternating_permission and not self.permission_mode:
            raise ValueError("PPO_ATTACK_PHASES requires PPO_ATTACK_PERMISSION=1")
        self.permission_phase_steps = 0
        self.permission_phase_index = 0
        self.permission_no_depth_steps = 0
        self.permission_phase_progress = 0.0
        self.permission_phase_stationary_steps = 0
        # CB1 policy controls: move, jump, vertical move, crouch, look X,
        # look Y, and attack. Decode() supplies inactive defaults for the
        # retained legacy control fields and appends a decision sequence.
        self.action_space = gym.spaces.MultiDiscrete(
            [3, 2, 3, 2, 9, 11, 2]
        )
        self.observation_space = gym.spaces.Dict({
            "general": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(
                    _general_input_dim
                    + int(self.permission_mode)
                    + int(self.attack_budget_mode),
                ),
                dtype=np.float32,
            ),
            "spatial": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(
                    ObsNorm.SPATIAL_CHANNELS,
                    Types.SIGHT_RANGE_X * 2 + 1,
                    Types.SIGHT_RANGE_Y * 2 + 1,
                ),
                dtype=np.float32,
            ),
        })

        self.latest_obs = None
        self.latest_observation_id = None
        self.last_consumed_observation_id = None
        self.obs_ready = threading.Event()
        self.previous_progress = None
        self.last_reward_terms = {}

        # Training episodes normally end on completion. These safeguards keep
        # one pathological procedural cave from dominating PPO with hundreds
        # of correlated rollouts. An incomplete watchdog ending is a genuine
        # failure for this descent-only policy, so it terminates without value
        # bootstrapping and receives Reward.INCOMPLETE_EPISODE_PENALTY.
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.max_stagnant_steps = MAX_STAGNANT_STEPS
        self.max_inactive_steps = MAX_INACTIVE_STEPS
        self.max_mobile_stagnant_steps = MAX_MOBILE_STAGNANT_STEPS
        self.episode_steps = 0
        self.best_layer_depth = None
        self.last_best_depth_step = 0
        self.inactive_steps = 0
        self.mobile_stagnant_steps = 0
        self.previous_tile_position = None
        self.previous_block_health = None
        self.episode_number = 0
        self.reset_token = 0
        self.world_seed = None
        self.episode_trace = EpisodeTraceWriter()

    def ConfigureAttackCurriculum(self, run_dir):
        if not self.attack_curriculum_mode:
            return
        self.attack_curriculum_state_path = (
            Path(run_dir) / "attack_curriculum_state.json"
        )
        self.attack_curriculum_state = AttackPermission.load_curriculum(
            self.attack_curriculum_state_path
        )
        (
            self.attack_curriculum_duration,
            self.attack_exploration_temperature,
        ) = AttackPermission.curriculum_values(self.attack_curriculum_state)
        self.attack_exploration_temperature_applied = (
            self.attack_exploration_temperature
        )
        AttackPermission.save_curriculum(
            self.attack_curriculum_state_path,
            self.attack_curriculum_state,
        )

    def PermissionPhaseLimit(self):
        if self.attack_curriculum_mode and self.attack_forbidden:
            return self.attack_curriculum_duration
        return 150

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.permission_phase_steps = 0
        self.permission_phase_index = 0
        self.permission_no_depth_steps = 0
        self.permission_phase_progress = 0.0
        self.permission_phase_stationary_steps = 0
        if self.permission_mode:
            requested = (options or {}).get("attack_forbidden")
            if requested is not None and requested not in (0, 1):
                raise ValueError("attack_forbidden must be 0 or 1")
            self.attack_forbidden = bool(
                self.np_random.integers(0, 2) if requested is None else requested
            )
            if self.alternating_permission:
                self.attack_forbidden = False
            self.attack_cost = float(
                self.attack_forbidden and self.attack_penalty_mode
            )
        if self.attack_budget_mode:
            requested_budget = (options or {}).get("attack_budget")
            if requested_budget is None:
                self.attack_budget = ResourceBudget.sample_budget(
                    self.np_random
                )
            else:
                self.attack_budget = int(requested_budget)
                if not (
                    ResourceBudget.MIN_ATTACK_BUDGET
                    <= self.attack_budget
                    <= ResourceBudget.MAX_ATTACK_BUDGET
                ):
                    raise ValueError("attack_budget is outside the CB1 range")
            self.attack_budget_used = 0

        requested_world_seed = None
        if options is not None:
            requested_world_seed = options.get("world_seed")
        if requested_world_seed is None:
            requested_world_seed = seed
        if requested_world_seed is None:
            requested_world_seed = self.np_random.integers(
                MIN_WORLD_SEED,
                MAX_WORLD_SEED + 1,
            )
        self.world_seed = NormalizeWorldSeed(requested_world_seed)

        self.episode_number += 1
        self.episode_trace.begin_episode(self.episode_number, self.world_seed)
        self.episode_steps = 0

        with _observation_lock:
            self.obs_ready.clear()

        for reset_attempt in range(MAX_RESET_ATTEMPTS):
            self.reset_token += 1
            reset_token = self.reset_token
            SendReset(reset_token, self.world_seed)

            # The acknowledgement carries the ID of the first observation
            # emitted after Unity has completed this reset. An observation
            # with a newer ID than the previous episode is not sufficient:
            # the last old-world packet can arrive after RESET was sent.
            first_observation_id = _server.WaitForResetReady(
                reset_token,
                self.world_seed,
            )
            obs, observation_id = self._wait_for_observation_id(
                first_observation_id
            )
            if not bool(obs["PlayerDead"]):
                break

            # A reset acknowledgement is tied to one exact post-reset
            # observation. If that observation is unusable, issue a new
            # tokenized reset rather than accepting another ambiguous packet.
        else:
            raise RuntimeError(
                "Unity reset did not produce a live observation after "
                f"{MAX_RESET_ATTEMPTS} attempts"
            )

        _server.reset_requested.clear()

        self.last_consumed_observation_id = observation_id
        self.best_layer_depth = int(obs["BestLayerDepth"])
        self.last_best_depth_step = 0
        self.inactive_steps = 0
        self.mobile_stagnant_steps = 0
        self.previous_tile_position = (
            int(obs["PlayerTilePosition"]["X"]),
            int(obs["PlayerTilePosition"]["Y"]),
        )
        self.previous_block_health = obs["RelativeBlockMap"]["Health"].copy()
        Reward.Reset(self, obs)
        processed = PreprocessObservation(obs)
        ProtocolValidation.validate_reset_observation(
            obs, processed, natural_stamina=self.natural_stamina_mode
        )
        if self.permission_mode:
            processed = AttackPermission.condition(processed, self.attack_forbidden)
        if self.attack_budget_mode:
            processed = ResourceBudget.condition(
                processed, self.attack_budget - self.attack_budget_used
            )
        return processed, {"world_seed": self.world_seed,
                           "attack_forbidden": self.attack_forbidden,
                           "attack_budget": self.attack_budget}

    def _wait_for_observation_id(self, expected_id):
        """Return exactly the observation identified by expected_id."""
        while True:
            with _observation_lock:
                current_id = self.latest_observation_id
                if current_id == expected_id:
                    obs = self.latest_obs
                    self.obs_ready.clear()
                    return obs, current_id

                if current_id is not None and current_id > expected_id:
                    raise RuntimeError(
                        "CB1 observation ID mismatch: "
                        f"expected {expected_id}, observed {current_id}"
                    )

                self.obs_ready.clear()

            self.obs_ready.wait()

    def step(self, action):
        requested_action = np.array(action, copy=True)
        if self.permission_mode and not self.attack_penalty_mode:
            action = AttackPermission.execute_action(action, self.attack_forbidden)
        Decode(action)

        if self.last_consumed_observation_id is None:
            raise RuntimeError("CB1 Env.step called before Env.reset")

        obs, observation_id = self._wait_for_observation_id(
            self.last_consumed_observation_id + 1
        )

        self.episode_steps += 1

        self.last_consumed_observation_id = observation_id
        processed = PreprocessObservation(obs)
        validation_info = ProtocolValidation.validate_step_observation(
            obs,
            processed,
            action,
            natural_stamina=self.natural_stamina_mode,
        )
        reward = Reward.Reward(obs, action, self)
        if self.attack_budget_mode:
            self.attack_budget_used += int(
                self.last_reward_terms["attack_swings_executed"]
            )
        reached_completion = bool(obs["LayerProgress"] >= 1.0)
        attack_budget_exceeded = bool(
            self.attack_budget_mode
            and ResourceBudget.exceeded(
                self.attack_budget_used, self.attack_budget
            )
        )
        completed = reached_completion and not attack_budget_exceeded
        if reached_completion and attack_budget_exceeded:
            # A completion outside its assigned resource constraint is not a
            # successful completion. Remove the bonus before applying the
            # ordinary incomplete-task penalty below.
            completion_bonus = float(self.last_reward_terms["completion"])
            reward -= completion_bonus
            self.last_reward_terms["completion"] = 0.0
            self.last_reward_terms["reward"] = reward
        continuous_progress_improved = (
            float(self.last_reward_terms["progress_delta"]) > 0.0
        )
        self.mobile_stagnant_steps = AdvanceMobileStagnation(
            self.mobile_stagnant_steps,
            self.last_reward_terms["progress_delta"],
        )

        best_layer_depth = int(obs["BestLayerDepth"])
        depth_improved = best_layer_depth > self.best_layer_depth
        if depth_improved:
            self.best_layer_depth = best_layer_depth
            self.last_best_depth_step = self.episode_steps

        inactivity_status = InactivityStatus(
            obs,
            self.previous_tile_position,
            self.previous_block_health,
        )
        physically_inactive = PhysicallyInactive(inactivity_status)
        next_inactive_steps = AdvanceInactivity(
            self.inactive_steps,
            inactivity_status,
            depth_improved,
            count_incapacity=self.natural_stamina_mode,
        )
        inactivity_timer_advanced = next_inactive_steps > 0
        inactivity_timer_mode = None
        if inactivity_timer_advanced:
            inactivity_timer_mode = (
                "physical" if inactivity_status["capable"] else "incapacitated"
            )
        self.inactive_steps = next_inactive_steps
        self.previous_tile_position = (
            int(obs["PlayerTilePosition"]["X"]),
            int(obs["PlayerTilePosition"]["Y"]),
        )
        self.previous_block_health = obs["RelativeBlockMap"]["Health"].copy()

        steps_since_best_depth = self.episode_steps - self.last_best_depth_step
        permission_phase_limit = self.PermissionPhaseLimit()
        if self.alternating_permission:
            self.permission_phase_steps += 1
            self.permission_no_depth_steps = (
                0 if depth_improved else self.permission_no_depth_steps + 1
            )
            self.permission_phase_progress += float(self.last_reward_terms["progress_delta"])
            self.permission_phase_stationary_steps += int(inactivity_status["velocity_inactive"])
        truncation_reason = None
        if not completed:
            truncation_reason = TruncationReason(
                episode_steps=self.episode_steps,
                last_best_depth_step=self.last_best_depth_step,
                inactive_steps=self.inactive_steps,
                max_episode_steps=self.max_episode_steps,
                max_stagnant_steps=self.max_stagnant_steps,
                max_inactive_steps=self.max_inactive_steps,
                mobile_stagnant_steps=self.mobile_stagnant_steps,
                max_mobile_stagnant_steps=self.max_mobile_stagnant_steps,
            )
            if (
                truncation_reason == "capable_inactivity"
                and self.natural_stamina_mode
                and not inactivity_status["capable"]
            ):
                truncation_reason = "incapacitated_inactivity"
        if (not completed and self.permission_mode
                and (not self.attack_curriculum_mode
                     or permission_phase_limit >= 150)
                and AttackPermission.stalled_attempt(
                    self.attack_forbidden,
                    self.permission_no_depth_steps if self.alternating_permission else steps_since_best_depth)):
            truncation_reason = (
                "attack_prohibited_no_depth"
                if self.terminal_attack_mode
                else "attack_costly_no_depth"
                if self.attack_penalty_mode
                else "attack_forbidden_no_depth"
            )
        attack_violation = bool(
            not completed
            and self.terminal_attack_mode
            and self.attack_cost >= 1.0
            and int(self.last_reward_terms["attack_swings_executed"]) > 0
        )
        if attack_violation:
            truncation_reason = "attack_prohibited_swing"
        if attack_budget_exceeded:
            truncation_reason = "attack_budget_exceeded"
        watchdog_failed = truncation_reason is not None
        if watchdog_failed and (
            self.alternating_permission
            or not (self.permission_mode and self.attack_forbidden)
        ):
            reward = Reward.ApplyIncompleteEpisodePenalty(self)

        # A watchdog ending is endogenous task failure, not an exogenous time
        # limit. Report it as terminal so SB3 cannot bootstrap through it.
        terminated = completed or watchdog_failed
        truncated = False
        curriculum_event = None
        curriculum_survived = None
        if (
            self.attack_curriculum_mode
            and self.attack_forbidden
            and (terminated or self.permission_phase_steps >= permission_phase_limit)
        ):
            curriculum_survived = bool(
                completed
                or (
                    not watchdog_failed
                    and self.permission_phase_steps >= permission_phase_limit
                )
            )
            self.attack_curriculum_state, curriculum_event = (
                AttackPermission.record_curriculum_outcome(
                    self.attack_curriculum_state,
                    curriculum_survived,
                )
            )
            (
                next_duration,
                next_temperature,
            ) = AttackPermission.curriculum_values(
                self.attack_curriculum_state
            )
            if self.attack_curriculum_state_path is not None:
                AttackPermission.save_curriculum(
                    self.attack_curriculum_state_path,
                    self.attack_curriculum_state,
                )
        else:
            next_duration = self.attack_curriculum_duration
            next_temperature = self.attack_exploration_temperature
        info = self.last_reward_terms.copy()
        info["missed_attack_penalty_mode"] = (
            self.missed_attack_penalty_mode
        )
        info["attack_prior_mode"] = self.attack_prior_mode
        info["natural_stamina_mode"] = self.natural_stamina_mode
        if self.attack_budget_mode:
            info.update(
                attack_budget=self.attack_budget,
                attack_budget_used=self.attack_budget_used,
                attack_budget_remaining=max(
                    self.attack_budget - self.attack_budget_used, 0
                ),
                attack_budget_exceeded=attack_budget_exceeded,
            )
        if self.alternating_permission:
            info.update(permission_phase_index=self.permission_phase_index,
                        permission_phase_steps=self.permission_phase_steps,
                        permission_phase_progress=self.permission_phase_progress,
                        permission_phase_stationary_steps=self.permission_phase_stationary_steps,
                        permission_no_depth_steps=self.permission_no_depth_steps,
                        permission_phase_limit=permission_phase_limit,
                        permission_phase_finished=(
                            terminated
                            or self.permission_phase_steps >= permission_phase_limit
                        ),
                        curriculum_survived=curriculum_survived,
                        curriculum_event=curriculum_event,
                        curriculum_duration=self.attack_curriculum_duration,
                        curriculum_temperature=self.attack_exploration_temperature,
                        curriculum_temperature_applied=(
                            self.attack_exploration_temperature_applied
                        ),
                        curriculum_next_duration=next_duration,
                        curriculum_next_temperature=next_temperature)
        info.update(getattr(self, "evaluation_context", {}))
        if self.permission_mode:
            info.update(attack_forbidden=self.attack_forbidden,
                        attack_cost=self.attack_cost,
                        attack_penalty_mode=self.attack_penalty_mode,
                        terminal_attack_mode=self.terminal_attack_mode,
                        attack_violation=attack_violation,
                        forbidden_no_depth_limit=AttackPermission.FORBIDDEN_NO_DEPTH_STEPS,
                        requested_attack=int(requested_action[6]),
                        executed_attack=int(action[6]))
        info.update(validation_info)
        info.update({
            "episode_steps": self.episode_steps,
            "steps_since_best_depth": steps_since_best_depth,
            "inactive_steps": self.inactive_steps,
            "mobile_stagnant_steps": self.mobile_stagnant_steps,
            "max_mobile_stagnant_steps": self.max_mobile_stagnant_steps,
            "continuous_progress_improved": continuous_progress_improved,
            "mobile_stagnation_timer_advanced": (
                not continuous_progress_improved
            ),
            "inactivity_capable": inactivity_status["capable"],
            "inactivity_incapacitated": not inactivity_status["capable"],
            "inactivity_physical_gates_met": physically_inactive,
            "inactivity_tile_unchanged": inactivity_status["tile_unchanged"],
            "inactivity_velocity_inactive": inactivity_status["velocity_inactive"],
            "inactivity_block_health_unchanged": inactivity_status[
                "block_health_unchanged"
            ],
            "inactivity_depth_unchanged": not depth_improved,
            "inactivity_timer_advanced": inactivity_timer_advanced,
            "inactivity_timer_mode": inactivity_timer_mode,
            "inactivity_timer_advanced_while_incapacitated": (
                inactivity_timer_advanced and not inactivity_status["capable"]
            ),
            "watchdog_failed": watchdog_failed,
            "truncation_reason": truncation_reason,
        })
        self.episode_trace.record(
            self.episode_steps,
            action,
            obs,
            observation_id,
            reward,
            info,
            terminated,
            truncated,
        )
        if terminated or truncated:
            self.episode_trace.finish_episode(
                complete=completed
            )

        # Complete the old phase's action, reward, terminal check and trace
        # before supplying the next permission to the policy.
        if (
            self.alternating_permission
            and not terminated
            and self.permission_phase_steps >= permission_phase_limit
        ):
            self.attack_forbidden = not self.attack_forbidden
            self.attack_cost = float(
                self.attack_forbidden and self.attack_penalty_mode
            )
            self.permission_phase_steps = 0
            self.permission_phase_index += 1
            self.permission_no_depth_steps = 0
            self.permission_phase_progress = 0.0
            self.permission_phase_stationary_steps = 0
            self.attack_curriculum_duration = next_duration
            self.attack_exploration_temperature = next_temperature
        if self.permission_mode:
            processed = AttackPermission.condition(processed, self.attack_forbidden)
        if self.attack_budget_mode:
            processed = ResourceBudget.condition(
                processed, self.attack_budget - self.attack_budget_used
            )
        return processed, reward, terminated, truncated, info

    def close(self):
        self.episode_trace.close()
        if _server is not None:
            _server.Shutdown()
        
