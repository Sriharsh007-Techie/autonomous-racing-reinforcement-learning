# BPA III – Car Racing Challenge

**Group:** 27 · **Team:** Gryffindor  
**Member:** Atharva Bhorpe, Sahil Mane, Sriharsh Kulkarni

This repository contains our submission for the RLLBC Bonus Point Assignment III.

---

## 1) What we train: Algorithm overview

We use **Twin Delayed Deep Deterministic Policy Gradient (TD3)** for continuous control:

- **Twin Q-nets** (independent critics) to mitigate value overestimation.
- **Delayed policy updates** (actor update every 2 critic steps) for smoother learning.
- **Target policy smoothing** with clipped noise during target value computation.
- Replay-based off-policy training with large batches.

TD3 was selected over the provided DDPG baseline for its stability on high-dimensional, continuous racing tasks.

---

## 2) What the network sees: Effective state space

From the raw observation we build a compact feature vector:

- **Car state (first 16 raw values, later cleaned):** steering/velocities/torques etc.  
  We drop indices **[4..10]** (global position, yaw, RGB) to avoid leakage and enforce invariance.
- **Cones (80 items):** each mapped from `[used, x, y, sideL, sideR]` to  
  `[used, distance, sin(angle), cos(angle), sideL, sideR]`.
- **Center line (20 points):** for each point we compute  
  `distance, sin(direction), cos(direction), sin(tangent), cos(tangent)`.

All parts are concatenated and cleaned as described above. This is the **effective input** used by the networks.

---

## 3) What the network outputs: Action space

Continuous 3-D action:
1. **steer** ∈ [−1, 1]
2. **throttle** ∈ [−1, 1]
3. **brake** ∈ [−1, 1]

A small safety rule in `convert_action` enforces **mutual exclusivity**:
- if `brake > 0`, throttle is suppressed;
- if `throttle > 0.5`, brake is suppressed.

---

## 4) Neural architectures

**Actor (policy):** MLP `400 → 300 → tanh(action_dim)`  
**Critics (Q1, Q2):** MLPs with input = `[state, action]` → `400 → 300 → 1`  
Activations are ReLU in hidden layers; actions are rescaled to env bounds.

---

## 5) Training setup (key hyperparameters)

- **Buffer:** 3,000,000 transitions
- **Batch:** 512
- **Lrs:** actor `1e-4`, critics `3e-4`
- **Discount γ:** 0.99
- **Soft target τ:** 0.005
- **Policy delay:** 2
- **Warmup:** 50k random steps
- **Exploration:** linear decay of action noise **0.35 → 0.10** over 2M steps  
- **Target smoothing noise:** ~0.10 (clipped to 0.5)
- **Eval cadence:** every 10k steps; we log return, episode length, and an approximate average forward velocity

All logs are optionally sent to **Weights & Biases**.

---

## 6) What makes it competitive (design choices)

- **Clean observation engineering** (angles as sin/cos, distances, tangents; removal of global pose/RGB).
- **Action sanitization** (no throttle+brake conflict).
- **Large buffer + batch** for stable gradient estimates.
- **Measured exploration** (strong initially, then tapered) matched to long training budgets.
- **Velocity & episode diagnostics** to verify the agent learns sustained speed without incurring penalties.

---

## 7) Result (evaluation)

Using the configuration in this repo, our evaluation produced an episodic return of:

**→ 88.04**

This indicates high and consistent forward progress with minimal penalties.

---

## File map

- `training.py` — TD3 loop with schedules, evaluation, saving best/latest.
- `agent_interface.py` — `convert_obs`, `convert_action`, and the `Agent` (actor) class.

