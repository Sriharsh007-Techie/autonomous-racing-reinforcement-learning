"""
============================================
        TD3 Training Script (Improved)
============================================
Mods:
- Actor LR = 1e-4
- Batch size = 512
- Exploration noise 0.35 -> 0.10 (decayed over 3M steps)
- Throttle/brake exclusivity enforced in agent
- Extra wandb logs: avg velocity, episode length, % full episodes
"""

from __future__ import annotations
import os, time, random, collections
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from gymnasium.spaces import Box
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

from agent_interface import convert_obs, convert_action, Agent
from util import create_env, save_model


@dataclass
class Config:
    run_name: str = "TD3_FAST_VEL~17_26AUG"
    seed: int = 42
    use_deterministic: bool = True

    # tracking
    use_wandb: bool = True
    log_interval: int = 500
    wandb_project: str = "RLLBC_BPA3"
    wandb_team: str | None = None

    # training budget
    total_steps: int = 3_000_000
    warmup_steps: int = 50_000
    eval_interval: int = 10_000
    eval_episodes: int = 1

    # optimization
    critic_lr: float = 3e-4
    actor_lr: float = 1e-4
    batch_size: int = 512
    discount: float = 0.99
    tau: float = 0.005

    # noise
    expl_noise_init: float = 0.35
    expl_noise_final: float = 0.10
    target_policy_noise_init: float = 0.20
    target_policy_noise_final: float = 0.10
    noise_decay_steps: int = 3_000_000
    noise_limit: float = 0.5
    policy_update_delay: int = 2

    # replay buffer
    buffer_capacity: int = 3_000_000

    # saving
    save_best: str = os.path.join("models", "model.obj")
    save_latest: str = os.path.join("models", "model_latest.obj")


# Critic Net
class CriticNet(nn.Module):
    def __init__(self, env, obs_dim: int):
        super().__init__()
        act_dim = int(np.prod(env.action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def linear_schedule(step, start_step, end_step, start_val, end_val):
    if step <= start_step:
        return start_val
    if step >= end_step:
        return end_val
    alpha = (step - start_step) / max(1, (end_step - start_step))
    return (1 - alpha) * start_val + alpha * end_val


if __name__ == "__main__":
    args = Config()
    run_id = f"{args.run_name}_{args.seed}_{int(time.time())}"

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_team, config=vars(args), name=run_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.use_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    train_env = create_env(args.seed, render_env=False, limit_speed_factor=None, render_width=1280)
    obs_dim = int(convert_obs(train_env.reset()[0]).shape[0])
    eval_env = create_env(args.seed, render_env=False, limit_speed_factor=None, render_width=1280)

    # networks
    criticA, criticB = CriticNet(train_env, obs_dim).to(device), CriticNet(train_env, obs_dim).to(device)
    criticA_tgt, criticB_tgt = CriticNet(train_env, obs_dim).to(device), CriticNet(train_env, obs_dim).to(device)
    actor, actor_tgt = Agent(train_env).to(device), Agent(train_env).to(device)

    actor_tgt.load_state_dict(actor.state_dict())
    criticA_tgt.load_state_dict(criticA.state_dict())
    criticB_tgt.load_state_dict(criticB.state_dict())

    critic_opt = optim.Adam(list(criticA.parameters()) + list(criticB.parameters()), lr=args.critic_lr)
    policy_opt = optim.Adam(actor.parameters(), lr=args.actor_lr)

    buffer = ReplayBuffer(
        args.buffer_capacity,
        Box(low=-np.inf * np.ones(obs_dim, dtype=np.float32), high=np.inf * np.ones(obs_dim, dtype=np.float32), dtype=np.float32),
        train_env.action_space, device, handle_timeout_termination=False
    )

    obs, _ = train_env.reset(seed=args.seed)
    obs = convert_obs(obs).to(device)
    reward_window = collections.deque(maxlen=50)
    best_score = -np.inf

    with tqdm(total=args.total_steps, desc="Training", ncols=140) as pbar:
        for step in range(args.total_steps):
            noise_std = linear_schedule(step, args.warmup_steps, args.warmup_steps + args.noise_decay_steps,
                                        args.expl_noise_init, args.expl_noise_final)

            if step < args.warmup_steps:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    act_raw = actor.get_action(obs)
                    noise = torch.randn_like(act_raw) * noise_std
                    act_raw = (act_raw + noise).clamp(-1, 1)
                    action = convert_action(act_raw)

            nxt_obs, reward, term, trunc, info = train_env.step(action)
            nxt_obs = convert_obs(nxt_obs).to(device)
            done = term or trunc
            buffer.add(obs.cpu(), nxt_obs.cpu(), action, float(reward), done, info)
            obs = nxt_obs if not done else convert_obs(train_env.reset()[0]).to(device)

            # =========================
            # Episode finished
            # =========================
            if done and "episode" in info:
                ep_r = float(info["episode"]["r"])
                ep_len = float(info["episode"]["l"])
                reward_window.append(ep_r)
                avg_recent = np.mean(reward_window)

                # --- NEW METRICS ---
                avg_velocity = (ep_r / ep_len) / 0.01 if ep_len > 0 else 0.0
                hit_full = 1.0 if ep_len >= 600 else 0.0

                pbar.set_postfix({
                    "R": f"{ep_r:.1f}",
                    "avg50": f"{avg_recent:.1f}",
                    "noise": f"{noise_std:.2f}",
                    "vel": f"{avg_velocity:.1f}"
                })

                if args.use_wandb:
                    wandb.log({
                        "ep_return": ep_r,
                        "avg50_return": avg_recent,
                        "expl_noise": float(noise_std),
                        "ep_len": ep_len,
                        "avg_velocity": avg_velocity,
                        "full_episode": hit_full
                    }, commit=False)

            # =========================
            # Learning step
            # =========================
            if step > args.warmup_steps:
                batch = buffer.sample(args.batch_size)

                with torch.no_grad():
                    nxt_act = actor_tgt.get_action(batch.next_observations)
                    t_noise = (torch.randn_like(nxt_act) * 0.1).clamp(-args.noise_limit, args.noise_limit)
                    nxt_act = (nxt_act + t_noise).clamp(-1, 1)

                    q1_tgt = criticA_tgt(batch.next_observations, nxt_act)
                    q2_tgt = criticB_tgt(batch.next_observations, nxt_act)
                    y = batch.rewards.flatten() + (1 - batch.dones.flatten()) * args.discount * torch.min(q1_tgt, q2_tgt).view(-1)

                q1_loss = F.mse_loss(criticA(batch.observations, batch.actions).view(-1), y)
                q2_loss = F.mse_loss(criticB(batch.observations, batch.actions).view(-1), y)
                critic_loss = q1_loss + q2_loss
                critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                if step % args.policy_update_delay == 0:
                    policy_loss = -criticA(batch.observations, actor.get_action(batch.observations)).mean()
                    policy_opt.zero_grad(); policy_loss.backward(); policy_opt.step()

                    for net, net_tgt in [(actor, actor_tgt), (criticA, criticA_tgt), (criticB, criticB_tgt)]:
                        for p, tp in zip(net.parameters(), net_tgt.parameters()):
                            tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

                    if args.use_wandb and step % args.log_interval == 0:
                        wandb.log({"actor_loss": policy_loss.item()}, commit=False)
                if args.use_wandb and step % args.log_interval == 0:
                    wandb.log({"critic_loss": critic_loss.item()}, commit=True)

            # =========================
            # Evaluation
            # =========================
            if step > args.warmup_steps and step % args.eval_interval == 0:
                total_r, total_len = 0, 0
                for _ in range(args.eval_episodes):
                    ob_eval, _ = eval_env.reset(seed=True)
                    ob_eval = convert_obs(ob_eval).to(device)
                    d = False
                    steps_eval = 0
                    ep_eval_r = 0.0
                    while not d:
                        with torch.no_grad():
                            act_eval = convert_action(actor.get_action(ob_eval))
                        ob_eval, r, term, trunc, _ = eval_env.step(act_eval)
                        ob_eval = convert_obs(ob_eval).to(device)
                        d = term or trunc
                        ep_eval_r += float(r)
                        steps_eval += 1
                    total_r += ep_eval_r
                    total_len += steps_eval

                avg_r = total_r / args.eval_episodes
                avg_steps = total_len / args.eval_episodes
                avg_velocity_eval = (avg_r / avg_steps) / 0.01 if avg_steps > 0 else 0.0

                print(f"[Eval] reward={avg_r:.2f}, steps={avg_steps:.1f}, vel={avg_velocity_eval:.1f}")
                if args.use_wandb:
                    wandb.log({
                        "eval_reward": avg_r,
                        "eval_steps": avg_steps,
                        "eval_avg_velocity": avg_velocity_eval
                    }, step=step, commit=False)

                if avg_r > best_score:
                    best_score = avg_r; save_model(actor, args.save_best)
                save_model(actor, args.save_latest)

            pbar.update(1)

    train_env.close(); eval_env.close()
