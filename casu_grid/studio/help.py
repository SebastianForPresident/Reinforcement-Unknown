"""One glossary for the whole application: tooltips, info buttons and the Help page read from here, so wording stays consistent.

Each topic has a ``short`` sentence (tooltips, chart captions) and a ``body`` (info dialog / Help page). The texts describe what this
codebase actually does (64 × 64 rollout, ten physics ticks per decision, arrival within one tile, ...), not generic RL folklore.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    key: str
    title: str
    short: str
    body: str


def _t(key, title, short, body):
    return Topic(key, title, short, body)


TOPICS: dict[str, Topic] = {t.key: t for t in [
    _t("environment", "Environment (world)", "One simulated 128 × 128 world with its own terrain and its own copy of the agent.",
       "An environment is one independent simulated world: its own destructible terrain, its own agent and its own goal. XY1 runs 64 of them in parallel "
       "so that the policy gathers 64 decisions at a time. Digging in one world never changes another."),
    _t("decision", "Policy decision", "One action chosen by the policy and held for ten physics ticks (0.2 s of simulated time).",
       "The policy looks at the world, chooses one value for each of seven action heads (move x, jump, move y, crouch, look dx, look dy, attack) and "
       "that choice is held for ten physics ticks of 20 ms. Throughput and budgets in this application are counted in decisions across all worlds."),
    _t("update", "PPO update", "One cycle: collect 64 decisions in every world, then improve the policy from that data.",
       "A PPO update first collects a rollout (64 decisions in each world, so 4,096 decisions with 64 worlds), then optimises the policy for 2 epochs "
       "over minibatches of 32 decisions, which is 256 gradient steps. After every update the latest checkpoint is rewritten."),
    _t("throughput", "End-to-end vs rollout throughput", "End-to-end counts the optimisation time as well; rollout only counts data collection.",
       "Rollout throughput = decisions collected per second of simulation. End-to-end throughput = the same decisions divided by collection time PLUS the "
       "time the GPU spends optimising the policy, so it is always lower. The end-to-end figure is the one that determines how long a training run takes."),
    _t("arrival", "Arrival", "The agent counts as having reached the goal when it is on the goal tile or one of its eight neighbours.",
       "An episode ends in arrival when the agent's tile is within one tile (including diagonals) of the goal tile. The agent then receives the arrival "
       "bonus and a new episode with a new random goal starts."),
    _t("truncation", "Truncation (timeout)", "An episode cut off because it ran out of its decision budget before arriving.",
       "Training episodes may last up to 100,000 decisions, so during early training almost no episode ends by timeout and the episodes that DO finish "
       "are mostly the easy ones. Evaluation uses a much shorter, fixed horizon so that unfinished episodes are counted as failures."),
    _t("arrivals_per_1k", "Arrivals per 1,000 decisions", "Goals reached per 1,000 decisions across all worlds; unaffected by how long episodes last.",
       "The most direct sign of learning during training. Because episodes are long, 'arrival rate among finished episodes' can be misleading (only finished "
       "episodes count). Arrivals per 1,000 decisions counts every arrival against every decision taken, finished episodes or not."),
    _t("arrival_rate", "Arrival rate among finished episodes", "Of the episodes that ended in this update, the share that ended by reaching the goal.",
       "Undefined (shown as a gap, never as zero) when no episode ended in the update. With the 100,000-decision ceiling nearly every finished episode is an "
       "arrival, so this number stays near 100 % and says little. Prefer arrivals per 1,000 decisions."),
    _t("progress", "Progress toward the goal", "How much of the starting distance the agent has closed at its best, for episodes still in progress.",
       "For every episode that is still running at the end of the update: 1 − (closest distance reached so far ÷ starting distance). 0 means no progress, "
       "1 means the goal was reached. Unlike arrivals this also rewards partial progress, so it moves earlier than the success rate."),
    _t("distance", "Distance to goal", "Distance in tiles between the agent and the goal, averaged over episodes in progress.",
       "'Current' is where the agents are at the end of the update; 'closest so far' is the best any agent in that episode has ever achieved. A policy that "
       "gets near but then wanders away shows a big gap between the two."),
    _t("decisions_to_arrival", "Decisions to arrival", "How many decisions the episodes that reached their goal needed.",
       "Only episodes that ended in arrival are counted, so the value is missing in updates without any arrival. Lower is better; compare it with the "
       "starting distance because far goals take longer."),
    _t("out_of_world", "Agents outside the world", "Share of running episodes whose agent has left the 128 × 128 world, usually by digging through the floor and falling.",
       "The floor of an XY1 world can be destroyed and nothing ends an episode when the agent falls out of the world, so a fallen agent keeps falling until the "
       "100,000-decision ceiling. Every such agent is lost to training. Rising values mean fewer and fewer agents can learn anything, whatever the policy does. "
       "This is a property of the experiment as defined (the Studio does not change it); distance statistics therefore count only agents still inside the world."),
    _t("excavation", "Excavation", "Blocks the agent destroyed by attacking, counted per 1,000 decisions.",
       "Counts blocks actually destroyed (not attack presses). Digging is an exploratory behaviour in XY1; whether it helps navigation is part of what "
       "the experiment studies."),
    _t("attack_rate", "Attack frequency", "Share of decisions in which the policy chose to attack.",
       "A random policy attacks about half of the time. A policy that learned digging is rarely useful everywhere will move away from 50 %."),
    _t("entropy", "Policy entropy", "How random the policy still is. High = exploring; falling steadily = becoming decisive.",
       "Sum of the entropies of the seven action heads, in nats. A fresh network is close to the maximum (about 8.8 here). A slow decline is normal learning; "
       "a collapse to near zero within a few updates means the policy stopped exploring."),
    _t("kl", "Approximate KL divergence", "How far the policy moved away from the one that collected the data, during this update.",
       "Typical healthy values are 0.001 to 0.02. Updates stop early when it exceeds the target (0.03). Persistently near zero = tiny learning steps; "
       "repeatedly hitting the target = steps that are too large."),
    _t("clip_fraction", "Clip fraction", "Share of training samples whose probability ratio hit the PPO clip limit (0.2).",
       "Near 0 means the policy hardly changed; above about 0.3 means the updates are aggressive. It is a health indicator of the optimiser, not a measure of "
       "navigation skill."),
    _t("policy_loss", "Policy loss", "The PPO surrogate objective that is minimised. Its value alone does not indicate how good the policy is.",
       "Because advantages are normalised, the policy loss hovers around zero. Look for NaN/infinite values (a failure) rather than a trend."),
    _t("value_loss", "Value loss", "How wrong the critic's return predictions are (smaller = better predictions).",
       "Rises when rewards become larger or more varied (for example the first arrivals). A spike is not necessarily bad; compare with explained variance."),
    _t("explained_variance", "Explained variance", "How much of the variation in returns the critic explains: 1 = perfectly, 0 or below = no better than the average.",
       "A critic that explains nothing (≈ 0) makes learning slow and noisy. Rising explained variance together with rising arrivals is the usual picture of "
       "a policy that is genuinely learning."),
    _t("grad_norm", "Gradient norm", "Size of the gradient before clipping (clip limit 0.5).",
       "Values far above the clip limit mean clipping is doing most of the work; sudden growth often precedes instability."),
    _t("learning_rate", "Learning rate", "Step size of the optimiser (fixed at 0.0003 in XY1).", "Constant for the whole run in XY1; this application never changes it."),
    _t("reward", "Reward components", "XY1 pays 0.1 × every new best distance improvement plus a 10-point arrival bonus.",
       "Mean per-decision reward contributions. 'distance_improvement' is paid whenever the agent gets closer to the goal than ever before in the episode; "
       "'arrival_bonus' is paid once on arrival. Returns rise with arrivals but also with merely approaching the goal, which is why reward alone does not prove reliable navigation."),
    _t("return", "Episode return", "Total reward of episodes that finished in this update.", "Missing when no episode finished. Biased toward quick, easy episodes while training episodes are long."),
    _t("checkpoint", "Checkpoint", "A saved copy of the policy, optimiser and environment state that training can continue from.",
       "latest.pt is rewritten after every update (atomically, so an interruption never leaves a broken file). Numbered checkpoints are kept every N updates, and "
       "checkpoints/initial.pt is the untrained policy, the baseline for evaluation."),
    _t("resume", "Resuming a run", "Continues from a checkpoint in a NEW run folder; the update counter keeps counting.",
       "Resume restores the network, optimiser, random-number state and every world (terrain edits, goals, positions). It is only offered for checkpoints whose "
       "learning contract, world count and worker layout match the current configuration. Resumed runs are not bit-identical to uninterrupted ones "
       "(physics contact caches restart) but are statistically equivalent."),
    _t("evaluation", "Held-out evaluation", "Scores a policy on terrains it has never trained on, with identical goals for every policy compared.",
       "Evaluation uses terrains from a seed range far outside the training range, identical goal draws for every policy, a fixed decision horizon and no learning. "
       "Success = reached the goal within the horizon. Deterministic mode uses the most likely action; stochastic mode samples like training."),
    _t("training_vs_eval", "Training vs evaluation", "Training statistics come from a changing, exploring policy on training terrains; evaluation is controlled.",
       "An improving training curve can come from memorising the 64 training terrains. Only the held-out evaluation, with an untrained baseline under the same "
       "conditions, shows whether skill generalises."),
    _t("learning_curve", "What does a learning curve establish?", "That a number changed during training, not that the agent navigates reliably.",
       "A rising reward or falling loss alone is not evidence of reliable navigation. Check arrivals per 1,000 decisions and distance progress, "
       "confirm with held-out evaluation against the untrained baseline, and look at several seeds before drawing conclusions."),
    _t("worlds", "Worlds and workers", "64 worlds are spread over 8 worker processes with 2 PhysX threads each (the validated layout).",
       "Each worker process simulates 8 worlds on the CPU; the policy network runs on the GPU. More workers than CPU cores, or fewer than 2 PhysX threads per "
       "worker, is slower or hangs, so these are validated defaults rather than free settings."),
    _t("legacy_spawn", "Legacy spawn mode", "A historical mode that mis-places every world except the first. Never used for real training.",
       "Before Optimization IV the XY1 mini-world placed worlds 1–63 at the wrong position. The flag CASU_LEGACY_MINI_SPAWN=1 reproduces that for historical "
       "comparison only; the Studio refuses to train or evaluate with it."),
]}


def short(key: str) -> str:
    topic = TOPICS.get(key)
    return topic.short if topic else ""


def title(key: str) -> str:
    topic = TOPICS.get(key)
    return topic.title if topic else key


def body(key: str) -> str:
    topic = TOPICS.get(key)
    return topic.body if topic else ""
