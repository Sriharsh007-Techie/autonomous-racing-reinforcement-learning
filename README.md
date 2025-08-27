
# BPA III – Car Racing Challenge

**Group:** 27 **Team:** Gryffindor  
**Member:** Sriharsh Kulkarni, Sahil Mane & Atharva Bhorpe

This repository contains our final submission for the **RLLBC Bonus Point Assignment III**.

---

## 1) What we train: Algorithm overview

We use **Twin Delayed Deep Deterministic Policy Gradient (TD3)** for continuous control in the racing environment:

- **Twin Q-networks** (independent critics) to reduce Q-value overestimation.
- **Delayed actor updates** (policy updated every 2 critic updates) for stability.
- **Target policy smoothing** by adding clipped noise during target estimation.
- **Replay-based off-policy learning** to leverage past experiences efficiently.

TD3 is preferred over the baseline DDPG because of its **better stability and performance** on high-dimensional, continuous action tasks like racing.

---

## 2) What the network sees: Effective state space

From the raw environment observation, we engineer a compact, invariant feature vector:

- **Car state (first 16 values):** steering, velocities, torques, and other dynamic parameters.  
  We remove indices `[4..10]` (position, yaw, RGB, distance) to prevent data leakage.
- **Cones (80 items):** each `[used, x, y, sideL, sideR]` is transformed into  
  `[used, distance, sin(angle), cos(angle), sideL, sideR]`.
- **Center line (20 points):** for each point we derive  
  `[distance, sin(direction), cos(direction), sin(tangent), cos(tangent)]`.

These components are concatenated to create the **final input vector** used by both actor and critic networks.

---

## 3) What the network outputs: Action space

Continuous **3-dimensional action vector**:

1. **Steer** ∈ [−1, 1]  
2. **Throttle** ∈ [−1, 1]  
3. **Brake** ∈ [−1, 1]

The `convert_action` function ensures:
- Output clipping to `[-1, 1]`
- Proper dimensionality handling for compatibility with the environment

---

## 4) Neural architectures

**Actor (policy network):**
- Input: processed observation vector  
- Hidden layers: `64 → 128` with ReLU activation  
- Output: `tanh` scaled to match environment action bounds  

**Critic (Q1, Q2):**
- Input: concatenation of `[state, action]`  
- Hidden layers: `64 → 128` with ReLU activation  
- Output: single Q-value estimate  

---

## 5) Training setup (key hyperparameters)

- **Buffer size:** 2,000,000 transitions  
- **Batch size:** 216  
- **Learning rates:** actor `6e-4`, critic `6e-4`  
- **Discount factor γ:** 0.99  
- **Soft target update τ:** 0.005  
- **Policy update frequency:** every 2 critic updates  
- **Exploration noise:** fixed Gaussian noise `0.2`  
- **Target noise:** `0.2` clipped to `0.5`  
- **Warmup:** 25k random steps before policy learning  
- **Eval cadence:** every 10k steps with logging and model saving  

---

## 6) What makes it robust

- **Clean observation engineering** — sin/cos encoding and distance-based features for rotational invariance.  
- **Twin critic setup** — stable Q-learning with reduced bias.  
- **Replay buffer** — large capacity ensures diverse training samples.  
- **Noise injection** — promotes exploration while keeping control stable.  
- **W&B logging** — enables tracking actor/critic losses, evaluation rewards, and episode lengths.  

---

## 7) Results (evaluation)

With the current setup:
- **Average episodic return:** ~80–85 during late training  
- **Consistent policy behavior:** smooth driving and steady velocity without excessive penalties  

Further tuning (exploration decay and larger networks) is expected to push rewards even higher.

---

## 8) File map

- `training.py` — TD3 training loop, replay buffer, evaluation, and model saving.  
- `agent_interface.py` — `convert_obs`, `convert_action`, and `Agent` (actor) network for evaluation.  
- `models/` — stores checkpoints for best and last models after training.  

---

## 9) Next steps

To further improve performance, the following can be explored:
- Gradual decay of exploration noise.  
- Larger network sizes (`400-300` layers) for better function approximation.  
- Velocity diagnostics in logging to correlate reward improvements with speed control.  
- Multi-seed training to find more consistent policies.  
```
