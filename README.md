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
