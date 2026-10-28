# RoboColosseum — VLA Fine-Tuning and Evaluation

RoboColosseum brings together robot-specific workflows for adapting and evaluating vision-language-action (VLA) models on real robots. This repository family covers data preparation, fine-tuning, model deployment, and evaluation across Unitree G1, Franka, and bimanual YAM platforms.

The projects integrate **GR00T, π0.5 (PI05), MolmoAct2, LingBot-VLA, and Galaxea G0.5 (G05)** with robot-specific observations, action representations, and control interfaces. Each repository provides the code and configuration for its part of the workflow; available features and setup requirements vary by platform.

## Repositories

| Repository | Robot platform | Current focus |
|---|---|---|
| [robocolosseum-finetune-g1](https://github.com/RoboColosseum/robocolosseum-finetune-g1) | Unitree G1 | Multi-model deployment, web console, WBC integration, and rollout recording |
| [robocolosseum-finetune-franka](https://github.com/RoboColosseum/robocolosseum-finetune-franka) | Franka / DROID | Interactive evaluation, trial scoring, session recovery, and LeRobot export |
| [robocolosseum-finetune-yam](https://github.com/RoboColosseum/robocolosseum-finetune-yam) | Bimanual YAM | Dataset preparation, per-model fine-tuning configurations, and training launchers |

## This Repository: G1

This repository deploys fine-tuned policies on Unitree G1 through the WBC control interface. It includes a catalog of 25 model profiles across five model families and five tasks, a web console for model and client lifecycle management, recorded-observation inference tests, and rollout recording with an optional LeRobot export tool.

Model weights, recorded datasets, and machine-specific configuration are stored separately from the source code. Environment definitions, configuration templates, and external dependency metadata are included in the repository.
