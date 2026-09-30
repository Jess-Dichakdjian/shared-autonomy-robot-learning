# Uncertainty-Aware Shared Autonomy for Robotic Manipulation

**MSc Thesis Research — Politecnico di Milano | Visiting Research at University College London (UCL)**

> **Status:** Active research project — implementation in progress.

This repository documents my MSc thesis research on **uncertainty-aware shared autonomy for robotic manipulation**, combining **Vision-Language-Action models, diffusion policies, uncertainty estimation, and human corrective demonstrations**.

The research is being conducted at **University College London (UCL)** as part of my MSc in **Robotics and Mechatronics Engineering at Politecnico di Milano**.

### Current Research Stage

At the current stage of the project:

- literature review and research pipeline definition are complete
- initial diffusion-policy setup and small-scale testing have been performed
- VLA model integration and evaluation are currently in progress
- the project builds on an existing UCL MuJoCo manipulation benchmark using a Franka robot
- XR-based human corrective demonstrations and uncertainty-aware shared autonomy are planned for later stages

No final experimental results are available yet.

## Research Problem

Robot learning policies can perform complex manipulation tasks, but their predictions may become unreliable when the robot encounters unfamiliar states, ambiguous observations, or situations outside the data used to train the policy.

This thesis investigates how a robot can identify such situations and request targeted human assistance rather than relying entirely on either autonomous control or continuous teleoperation.

## Research Objective

The objective is to develop and evaluate a shared-autonomy pipeline in which:

1. a learned manipulation policy generates robot actions,
2. uncertainty is estimated during task execution,
3. high-uncertainty situations trigger human intervention,
4. the operator provides corrective demonstrations through an XR interface,
5. the collected corrections can be used to improve subsequent robot behaviour.

The project is focused on **robotic manipulation with a Franka manipulator in MuJoCo**, with the broader goal of studying how human corrections can be incorporated efficiently into robot-learning systems.


