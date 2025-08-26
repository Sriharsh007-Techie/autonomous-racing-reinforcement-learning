"""
============================================
        Reinforcement Learning Trainer
============================================
Custom TD3 training on Racing-Env.
Bigger nets, noise decay, balanced LRs,
avg50 tracking, and progress bar with ETA.
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
from tqdm import tqdm   # progress bar

from V1_agent_interface import convert_obs, convert_action, Agent
from util import create_env, save_model


# =========================================================
# Configuration
# =========================================================
@dataclass
class Config:
    run_name: str = "TD3_2AUG2025_0051"
    seed: int = 42
    use_deterministic: bool = True

    # experiment tracking
    use_wandb: bool = True
    log_interval: int = 500
    wandb_project: str = "RLLBC_BPA3"
    wandb_team: str | None = None

    # training budget
    total_steps: int = 3_000_000
    warmup_steps: int = 50_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    render_eval: bool = False

    # optimization
    critic_lr: float = 3e-4
    actor_lr: float = 1e-4
    batch_size: int = 216
    discount: float = 0.99
    tau: float = 0.005

    # TD3 noise
    expl_noise_init: float = 0.25
    expl_noise_final: float = 0.05
    target_policy_noise_init: float = 0.20
    target_policy_noise_final: float = 0.05
    noise_decay_steps: int = 1_000_000
    noise_limit: float = 0.5
    policy_update_delay: int = 2

    # replay buffer
    buffer_capacity: int = 2_000_000

    # model saving
    save_best: str = os.path.join("models", "V1_td3_best.obj")
    save_latest: str = os.path.join("models", "V1_td3_latest.obj")


# =========================================================
# Critic network (bigger: 256-256)
# =========================================================
class CriticNet(nn.Module):
    def __init__(self, env, obs_dim: int):
        super().__init__()
        act_dim = int(np.prod(env.action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =========================================================
# Utils
# =========================================================
def grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    num_elems = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data.norm(2)
            total_sq += float(g.item() ** 2)
            num_elems += p.grad.numel()
    return (total_sq ** 0.5) / max(1, num_elems)


def linear_schedule(step: int, start_step: int, end_step: int, start_val: float, end_val: float) -> float:
    if step <= start_step:
        return start_val
    if step >= end_step:
        return end_val
    alpha = (step - start_step) / max(1, (end_step - start_step))
    return (1 - alpha) * start_val + alpha * end_val


# =========================================================
# Training loop
# =========================================================
if __name__ == "__main__":
    args = Config()
    run_id = f"{args.run_name}_{args.seed}_{int(time.time())}"

    # logging
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_team,
            config=vars(args),
            name=run_id,
            monitor_gym=True,
            save_code=True,
        )

    # moving average tracker (50 episodes)
    reward_window = collections.deque(maxlen=50)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.use_deterministic

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # envs
    train_env = create_env(args.seed, render_env=False, limit_speed_factor=None, render_width=1280)
    obs_dim = int(convert_obs(train_env.reset()[0]).shape[0])
    eval_env = create_env(args.seed, render_env=args.render_eval, limit_speed_factor=None, render_width=1280)
    assert isinstance(eval_env.action_space, Box)

    # networks
    criticA, criticB = CriticNet(train_env, obs_dim).to(device), CriticNet(train_env, obs_dim).to(device)
    criticA_tgt, criticB_tgt = CriticNet(train_env, obs_dim).to(device), CriticNet(train_env, obs_dim).to(device)
    actor, actor_tgt = Agent(train_env).to(device), Agent(train_env).to(device)

    actor_tgt.load_state_dict(actor.state_dict())
    criticA_tgt.load_state_dict(criticA.state_dict())
    criticB_tgt.load_state_dict(criticB.state_dict())

    # optimizers
    critic_opt = optim.Adam(list(criticA.parameters()) + list(criticB.parameters()), lr=args.critic_lr)
    policy_opt = optim.Adam(actor.parameters(), lr=args.actor_lr)

    # replay buffer
    proc_obs_space = Box(
        low=-np.inf * np.ones(obs_dim, dtype=np.float32),
        high=np.inf * np.ones(obs_dim, dtype=np.float32),
        dtype=np.float32,
    )
    train_env.observation_space.dtype = np.float32
    buffer = ReplayBuffer(args.buffer_capacity, proc_obs_space, train_env.action_space, device, handle_timeout_termination=False)

    # loop vars
    obs, _ = train_env.reset(seed=args.seed)
    obs = convert_obs(obs).to(device)
    best_score = -np.inf

    # =========================
    # Progress bar wrapper (with ETA)
    # =========================
    with tqdm(total=args.total_steps, desc="Training", ncols=140, dynamic_ncols=True) as pbar:
        for step in range(args.total_steps):
            # ---- noise decay ----
            noise_std = linear_schedule(
                step, args.warmup_steps, args.warmup_steps + args.noise_decay_steps,
                args.expl_noise_init, args.expl_noise_final
            )
            target_noise_std = linear_schedule(
                step, args.warmup_steps, args.warmup_steps + args.noise_decay_steps,
                args.target_policy_noise_init, args.target_policy_noise_final
            )

            # --- pick action ---
            if step < args.warmup_steps:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    act_raw = actor.get_action(obs)
                    noise = torch.randn_like(act_raw) * noise_std
                    act_raw = (act_raw + noise).clamp(-1, 1)
                    action = convert_action(act_raw)

            # --- env step ---
            nxt_obs, reward, term, trunc, info = train_env.step(action)
            reward = float(reward)
            nxt_obs = convert_obs(nxt_obs).to(device)
            done = term or trunc
            buffer.add(obs.cpu(), nxt_obs.cpu(), action, reward, done, info)
            obs = nxt_obs if not done else convert_obs(train_env.reset()[0]).to(device)

            if done and "episode" in info:
                ep_r = float(info["episode"]["r"])
                reward_window.append(ep_r)
                avg_recent = float(np.mean(reward_window)) if len(reward_window) > 0 else ep_r
                pbar.set_postfix({
                    "R": f"{ep_r:.1f}",
                    "avg50": f"{avg_recent:.1f}",
                    "noise": f"{float(noise_std):.2f}"
                })
                if args.use_wandb:
                    wandb.log(
                        {"ep_return": ep_r, "avg50_return": avg_recent, "expl_noise": float(noise_std)},
                        commit=False,
                    )

            # --- learning ---
            if step > args.warmup_steps:
                batch = buffer.sample(args.batch_size)

                with torch.no_grad():
                    nxt_act = actor_tgt.get_action(batch.next_observations)
                    t_noise = (torch.randn_like(nxt_act) * target_noise_std).clamp(-args.noise_limit, args.noise_limit)
                    nxt_act = (nxt_act + t_noise).clamp(-1, 1)

                    q1_tgt = criticA_tgt(batch.next_observations, nxt_act)
                    q2_tgt = criticB_tgt(batch.next_observations, nxt_act)
                    y = batch.rewards.flatten() + (1 - batch.dones.flatten()) * args.discount * torch.min(q1_tgt, q2_tgt).view(-1)

                q1_loss = F.mse_loss(criticA(batch.observations, batch.actions).view(-1), y)
                q2_loss = F.mse_loss(criticB(batch.observations, batch.actions).view(-1), y)
                critic_loss = q1_loss + q2_loss

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                if step % args.policy_update_delay == 0:
                    policy_loss = -criticA(batch.observations, actor.get_action(batch.observations)).mean()
                    policy_opt.zero_grad()
                    policy_loss.backward()
                    policy_opt.step()

                    for net, net_tgt in [(actor, actor_tgt), (criticA, criticA_tgt), (criticB, criticB_tgt)]:
                        for p, tp in zip(net.parameters(), net_tgt.parameters()):
                            tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

                    if args.use_wandb and step % args.log_interval == 0:
                        wandb.log({"actor_loss": policy_loss.item(), "grad_norm": grad_norm(actor)}, commit=False)
                if args.use_wandb and step % args.log_interval == 0:
                    wandb.log({"critic_loss": critic_loss.item()}, commit=True)

            # --- evaluation ---
            if step > args.warmup_steps and step % args.eval_interval == 0:
                total_r, total_len = 0, 0
                for _ in range(args.eval_episodes):
                    ob_eval, _ = eval_env.reset(seed=True)
                    ob_eval = convert_obs(ob_eval).to(device)
                    d = False
                    while not d:
                        with torch.no_grad():
                            act_eval = convert_action(actor.get_action(ob_eval))
                            act_eval = np.clip(act_eval, eval_env.action_space.low, eval_env.action_space.high)
                        ob_eval, r, term, trunc, info_eval = eval_env.step(act_eval)
                        r = float(r)
                        ob_eval = convert_obs(ob_eval).to(device)
                        d = term or trunc
                        total_r += r
                        total_len += 1
                avg_r = total_r / args.eval_episodes

                pbar.write(f"[Eval] reward={avg_r:.2f}, steps={total_len / args.eval_episodes:.1f}")

                if args.use_wandb:
                    wandb.log({"eval_reward": avg_r, "eval_steps": total_len}, step=step, commit=False)

                if avg_r > best_score:
                    best_score = avg_r
                    save_model(actor, args.save_best)
                save_model(actor, args.save_latest)

            pbar.update(1)  # update progress bar end of step

    train_env.close()
    eval_env.close()
