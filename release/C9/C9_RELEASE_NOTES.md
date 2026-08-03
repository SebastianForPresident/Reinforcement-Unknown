# Checkpoint 9

2026-08-03_10-54-33  
Steps completed: 600k checkpoint (635,821 trace steps recorded)  
Max episode steps: 25k  
Episode termination: Death or complete layer (layer progress >= 1)  
Total training time: approximately 4 hours  
Reward function: V7  
Latest commit: bed5614  

## Training statistics

- Episodes recorded: 34
- Mean episode length: 18,701 steps
- Episodes ending in death: 24
- Episodes reaching the time limit: 10
- Median best progress per episode: 3.12%
- Mean best progress per episode: 3.37%
- Best progress: 6.91%
- Best depth: 21
- Mean final TensorBoard episode reward: approximately -98.5

## Behavior summary

The policy did not complete a layer. Its dominant behavior was persistent exercise and ragdolling with very little meaningful descent.

- Exercise selected: 73.0% of recorded steps
- Ragdoll selected: 52.6% of recorded steps
- Stamina below 10: 25.8% of recorded steps
- Average pain above 80: 24.5% of recorded steps
- Shock above 60: 17.9% of recorded steps
- Critical blood pressure: 20.1% of recorded steps
- Critical bleed speed: 19.8% of recorded steps

Summary: C9 was a fresh V7 run intended to test whether stronger physiological penalties would force conservative behavior. It did not. The policy developed a persistent self-damaging attractor, repeatedly exercising and ragdolling while making little progress. The run was stopped after the 600k checkpoint.

Behavior preview: PPO_Test9.mp4
