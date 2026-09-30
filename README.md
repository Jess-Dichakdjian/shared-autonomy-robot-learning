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


## Planned System Pipeline

The current thesis is organised around the following **conceptual shared-autonomy pipeline**:

1. A **robot manipulation task** is defined in simulation  
   - current focus: **square manipulation**
   - **lift/can** tasks may be used as debugging baselines

2. The system receives **live environmental observations**, including:
   - simulated camera images
   - robot joint states

3. These observations feed **two parallel components**:

   **Primary Action Engine**
   - Diffusion Policy **or**
   - Vision-Language-Action (VLA) model

   **Secondary Estimator**
   - Gaussian Process Regression
   - used for uncertainty / variance tracking

4. The **primary action engine** produces robot actions for the control loop.

5. In parallel, the **secondary estimator** monitors whether the current state is associated with **high uncertainty** or possible **out-of-distribution (OOD)** behaviour.

6. A decision is then made:
   - if uncertainty is low, the system continues autonomous execution
   - if uncertainty is high, autonomy is paused and human intervention is requested

7. When intervention is needed, the operator provides a **corrective demonstration** through a **Unity XR interface**.

8. The corrective trajectory is collected and intended to support two stages of improvement:
   - **Stage 1:** real-time or near-real-time update of the uncertainty estimator
   - **Stage 2:** offline policy improvement / fine-tuning using corrective human demonstrations

### Pipeline Summary

**Task**  
Robot manipulation in simulation

**Observations**  
Camera images + robot joint states

**Parallel modules**  
- learned policy (Diffusion Policy / VLA)
- uncertainty estimator (Gaussian Process Regression)

**Decision logic**  
- low uncertainty → continue autonomy  
- high uncertainty / OOD → pause autonomy and request correction

**Human-in-the-loop correction**  
Unity XR interface for corrective demonstrations

**Learning from correction**  
- update uncertainty estimation
- improve future policy performance offline

## Current Implementation Status

| Component | Status | Current State |
|---|---|---|
| Literature review & system design | ✅ Completed | Research direction, system architecture and planned experimental pipeline defined |
| Diffusion Policy | 🟡 Initial testing | Existing implementation downloaded, configured and used for small-scale testing |
| VLA model | 🟡 In progress | Currently working on VLA integration and evaluation |
| MuJoCo manipulation environment | 🟡 Existing baseline / integration in progress | Thesis builds on an existing UCL Franka MuJoCo benchmark |
| Manipulation task | 🟡 In development | Square manipulation is the main target task; simpler tasks may be used for debugging |
| Gaussian Process uncertainty estimation | ⚪ Planned | Intended to provide uncertainty / variance estimates during policy execution |
| Uncertainty-triggered intervention | ⚪ Planned | Decision logic for requesting human correction has not yet been implemented |
| Unity / XR interface | ⚪ Planned | Will adapt XR teleoperation work developed during a previous robotics project |
| Human corrective demonstrations | ⚪ Planned | Corrective trajectory collection will be integrated later in the project |
| Online uncertainty-model update | ⚪ Planned | Intended to incorporate information from human corrective demonstrations |
| Offline policy improvement | ⚪ Planned | Corrective demonstrations are intended for later policy fine-tuning / improvement |
| Integrated shared-autonomy pipeline | ⚪ Planned | Full end-to-end system has not yet been implemented |
| Experimental evaluation | ⚪ Not started | No final quantitative results are available yet |

**Status key:**  
✅ Completed · 🟡 In progress / partially implemented · ⚪ Planned
