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



## Project Scope and Contribution

This thesis builds on an existing private UCL research repository, **`VLA_Benchmark`**, which provides baseline infrastructure for robot-learning experiments.

### Existing UCL Baseline

The upstream UCL repository already contains components including:

- a MuJoCo simulation environment
- a Franka manipulator model and simulation assets
- data-collection utilities
- existing VLA experimentation code
- infrastructure for connecting learned policies to the simulation

These components were **not originally developed by me**.

### My Thesis Work

My work focuses on extending this baseline toward an **uncertainty-aware shared-autonomy system**.

My current and planned contributions include:

- evaluating and integrating learned manipulation policies
- investigating Diffusion Policy and VLA approaches for action generation
- developing an uncertainty-estimation component based on Gaussian Process Regression
- designing logic for detecting high-uncertainty / out-of-distribution situations
- integrating an XR-based interface for human corrective demonstrations
- collecting corrective trajectories during human intervention
- investigating how corrective demonstrations can update the uncertainty model
- investigating offline policy improvement using collected corrections
- designing and evaluating the complete shared-autonomy pipeline

The distinction between **existing UCL infrastructure**, **third-party research code**, and **my own implementation** will be maintained throughout this repository.




## Technology Stack

### Currently Used

- **Python** — model experimentation and robotics research workflow
- **MuJoCo** — manipulation simulation
- **Franka Emika Panda** — target manipulator
- **Diffusion Policy** — initial policy experimentation
- **Vision-Language-Action Models** — current research and integration focus
- **Linux** — development environment
- **Git / GitHub** — version control and research workflow

### Planned Components

- **Gaussian Process Regression** — uncertainty estimation
- **Unity** — operator-side XR environment
- **XR / Meta Quest** — human corrective demonstrations
- **Human-in-the-loop learning** — intervention and corrective trajectory collection

> Technologies listed as planned are part of the thesis methodology but have not yet been fully implemented.

---

## Experimental Task

The primary manipulation task planned for the thesis is the **Square task**.

Simpler manipulation tasks such as **Lift** or **Can** may be used as debugging and validation baselines before evaluating the complete system on the target task.

The experimental setup is intended to evaluate both:

- autonomous manipulation performance
- the behaviour of the uncertainty-aware intervention mechanism

---

## Research Questions

The thesis investigates questions including:

1. Can uncertainty estimation identify situations in which a learned manipulation policy is likely to behave unreliably?
2. Can human intervention be requested selectively rather than requiring continuous teleoperation?
3. Can corrective demonstrations improve the system's understanding of uncertain states?
4. Can collected corrections subsequently improve manipulation-policy performance?
5. How do different action-generation approaches, such as Diffusion Policies and VLA models, behave within the shared-autonomy framework?

These questions define the current research direction and may be refined as implementation and experimentation progress.

---

## Evaluation Plan

The final evaluation methodology is still under development.

Planned evaluation areas include:

### Manipulation Performance

Potential measures include:

- task success rate
- task completion behaviour
- policy failures
- intervention frequency

### Uncertainty Estimation

The uncertainty estimator will be investigated in terms of its ability to distinguish between:

- familiar states
- uncertain states
- potentially out-of-distribution observations or behaviours

### Human Intervention

Corrective demonstrations will be evaluated based on factors such as:

- when intervention is triggered
- whether the correction allows task recovery
- the amount of human intervention required

### Learning from Corrections

Later experiments are intended to investigate whether collected human corrections can improve:

- uncertainty estimation
- future autonomous behaviour
- policy performance after offline updating or fine-tuning

> Exact metrics and experimental protocols will be added once the implementation and evaluation methodology are finalised.

---

## Repository Structure

This repository is being developed alongside the thesis and will expand as implementation progresses.

The intended structure is:

```text
shared-autonomy-robot-learning/
│
├── README.md
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── experiments.md
│
├── src/
│   ├── policies/
│   ├── uncertainty/
│   ├── shared_autonomy/
│   └── xr_interface/
│
├── scripts/
│
├── configs/
│
├── results/
│   ├── figures/
│   └── tables/
│
└── media/
    ├── diagrams/
    ├── screenshots/
    └── demos/
