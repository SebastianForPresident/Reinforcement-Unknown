# Reinforcement-Unknown
An attempt at shoving a PPO into Casualties: Unknown and seeing if it'll make it out alive on the other side.

## Overview

Reinforcement-Unknown is my attempt at training AI-powered NPCs in *Casualties: Unknown* using reinforcement learning (RL), beginning with Proximal Policy Optimization (PPO). It stemmed from my mod [Project Mechanism](https://www.nexusmods.com/scavprototype/mods/45) needing competent NPC behavior rather than structures holed up in pods

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
| Checkpoint 2 | V2 | Thrashing for mobility and terrain destruction |
| Checkpoint 3 | V3 | Lighter, non-destructive seizures |
| Checkpoint 4 | V4 | Wiggle to fall |
| Checkpoint 5 | V5 | Leave pod to try to descend |
| Checkpoint 6 | V5 | Exercise and avoid ledges + overall descent |
| Checkpoint 6.5 | V5 | Descend as much as possible |
| Checkpoint 7 | V6 | Unsafe, brute-force descent |
| Checkpoint 8 | V7 | Unproductive self-damage |
| Checkpoint 9 | V7 | Unproductive self-damage |
| Checkpoint 10 | V8 & V9 | Unproductive self-damage |
| Checkpoint 11 | V10 | Preserve health, stay safe, and stay stagnant |
| Checkpoint 12 | V10 | Minimal movement/exertion while trying to descend |
| Checkpoint 13 | V11, V12, & V13 | Blind descent and attack farming |
| Checkpoint 14 | V14 & V15 | Careful descent |
| Checkpoint 15 | V15 | High-exertion "descent" |
| Checkpoint 16 | V16 | Aggressive, high-entropy descent |