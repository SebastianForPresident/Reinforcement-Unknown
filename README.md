# Reinforcement-Unknown
An attempt at shoving a PPO into Casualties: Unknown and seeing if it'll make it out alive on the other side.

## Overview

Reinforcement-Unknown is my attempt at training reinforcement learning agents in *Casualties: Unknown* using standard RL models and techniques, such as Proximal Policy Optimization (PPO)

## Releases

Every GitHub Release represents a trained checkpoint.

Each release contains:

- Trained PPO checkpoint
- TensorBoard logs
- Matching plugin binary
- Training summary
- Behavior preview

## Goal

The long-term goal is to produce an agent capable of surviving and eventually completing an entire layer of Casualties: Unknown. 

The stretch goal is to complete multiple.

## Current Progress

| Checkpoint | Reward | Behavior |
|------------|--------|----------|
| Test | None | Random thrashing |
| Checkpoint 1 | V1 | Holds right, attacks upward, avoids ragdolling |
