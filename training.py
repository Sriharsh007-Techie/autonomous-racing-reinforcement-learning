"""
############################################
############# Training Script ##############
############################################
This code is part of a sample solution that can be a good starting point for how to structure
the training of an agent and do logging. However, the performance that this solution achieves
is not good.

Implementation based on https://docs.cleanrl.dev/rl-algorithms/ddpg/
"""

from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn, optim

from agent_interface import convert_action, convert_obs, Agent
from util import save_model, create_env


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def print_gradients(model: nn.Module, name: str) -> float:
    total_sq_norm = 0.0
    num_total_elems = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_sq_norm += float(param_norm.item() ** 2)
            num_total_elems += p.grad.numel()
    if num_total_elems == 0:
        return 0.0
    return (total_sq_norm ** 0.5) / num_total_elems


# -----------------------------------------------------------
# Args
# -----------------------------------------------------------

@dataclass
class Args:
    exp_name: str = "TD3_18_08_02"
    seed: int = 41
    torch_deterministic: bool = True
    track: bool = True
    track_frequency: int = 500
    wandb_project_name: str = "RLLBC_BPA3"
    wandb_entity: str | None = None
    save_model: bool = True

    # env
    env_id: str = "Racing-Env"

    # training schedule
    total_timesteps: int = 2_500_000
    learning_starts: int = 25_000
    eval_freq: int = 10_000
    num_eval_episodes: int = 1
    render_eval: bool = False

    # optimization
    learning_rate: float = 6e-4
    actor_learning_rate: float = 6e-4
    batch_size: int = 216
    gamma: float = 0.99
    tau: float = 0.005

    # exploration / TD3 tricks
    exploration_noise: float = 0.2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2

    # replay
    buffer_size: int = 2_000_000

    # saving
    best_model_save_path: str = os.path.join("models", "TD3_18_08_02_best.obj")
    last_model_save_path: str = os.path.join("models", "TD3_18_08_02_last.obj")


# -----------------------------------------------------------
# Networks
# -----------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, env, obs_shape: int):
        super().__init__()
        act_dim = int(np.prod(env.action_space.shape))
        self.fc1 = nn.Linear(obs_shape + act_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":
    args = Args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    # W&B
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("CUDA cache emptied.")
        print(f"CUDA device ID: {torch.cuda.current_device()}")
        print(f"CUDA device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # envs
    env = create_env(args.seed, render_env=False, limit_speed_factor=None, render_width=1280)
    obs_shape = int(convert_obs(env.reset()[0]).shape[0])

    env_eval = create_env(args.seed, render_env=args.render_eval, limit_speed_factor=None, render_width=1280)
    assert isinstance(env_eval.action_space, Box), "only continuous action space is supported"

    # networks
    qf1 = QNetwork(env, obs_shape).to(device)
    qf1_target = QNetwork(env, obs_shape).to(device)
    qf2 = QNetwork(env, obs_shape).to(device)
    qf2_target = QNetwork(env, obs_shape).to(device)
    actor = Agent(env).to(device)
    target_actor = Agent(env).to(device)

    # target init
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # optimizers
    qf_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.actor_learning_rate)

    # replay buffer (SB3)
    processed_obs_space = Box(
        low=-np.inf * np.ones(obs_shape, dtype=np.float32),
        high=np.inf * np.ones(obs_shape, dtype=np.float32),
        dtype=np.float32,
    )
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        processed_obs_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    # loop state
    global_step = 0
    data_device_checked = False
    obs, _ = env.reset(seed=args.seed)
    obs = convert_obs(obs).to(device)
    best_eval_reward = -np.inf

    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                raw_action = actor.get_action(obs)
                noise = torch.randn_like(raw_action) * args.exploration_noise
                raw_action = (raw_action + noise).clamp(-1, 1)
                action = convert_action(raw_action)
                action = np.clip(action, env.action_space.low, env.action_space.high)

        # Step env
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = convert_obs(next_obs).to(device)
        done = terminated or truncated

        rb.add(obs.cpu(), next_obs.cpu(), action, reward, done, info)

        if done:
            print(f"step={global_step}, episode_reward={info['episode']['r']}, length={info['episode']['l']}")
            if args.track:
                wandb.log({"episode_cumulative_reward": info["episode"]["r"], "episode_length": info["episode"]["l"]}, commit=False)
            next_obs, _ = env.reset()
            next_obs = convert_obs(next_obs).to(device)

        obs = next_obs

        # Learn
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            if not data_device_checked:
                print(f"Observations device: {data.observations.device}")
                print(f"Actions device: {data.actions.device}")
                data_device_checked = True

            with torch.no_grad():
                raw_next_action = target_actor.get_action(data.next_observations)
                noise = (torch.randn_like(raw_next_action) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                next_actions = (raw_next_action + noise).clamp(-1, 1)
                next_actions = next_actions * actor.action_scale + actor.action_bias

                q1_target = qf1_target(data.next_observations, next_actions)
                q2_target = qf2_target(data.next_observations, next_actions)
                min_q_target = torch.min(q1_target, q2_target)
                target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_q_target.view(-1)

            q1_loss = F.mse_loss(qf1(data.observations, data.actions).view(-1), target_q)
            q2_loss = F.mse_loss(qf2(data.observations, data.actions).view(-1), target_q)
            qf_loss = q1_loss + q2_loss

            qf_optimizer.zero_grad()
            qf_loss.backward()
            qf_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor.get_action(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # soft update
                for p, tp in zip(actor.parameters(), target_actor.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

                if args.track and global_step % args.track_frequency == 0:
                    wandb.log({"actor_loss": actor_loss.item(), "actor_grad": print_gradients(actor, "Actor")}, commit=False)
            if args.track and global_step % args.track_frequency == 0:
                wandb.log({"qf_loss": qf_loss.item()}, commit=True)

        # Evaluation
        if global_step > args.learning_starts and global_step % args.eval_freq == 0:
            total_reward, total_steps = 0, 0
            for i in range(args.num_eval_episodes):
                obs_eval, _ = env_eval.reset(seed=True)
                obs_eval = convert_obs(obs_eval).to(device)
                done, steps = False, 0
                while not done:
                    with torch.no_grad():
                        action_eval = convert_action(actor.get_action(obs_eval))
                        action_eval = np.clip(action_eval, env.action_space.low, env.action_space.high)
                    obs_eval, reward_eval, terminated, truncated, info_eval = env_eval.step(action_eval)
                    obs_eval = convert_obs(obs_eval).to(device)
                    done = terminated or truncated
                    total_reward += reward_eval
                    steps += 1
                total_steps += steps

            avg_reward = total_reward / args.num_eval_episodes
            avg_steps = total_steps / args.num_eval_episodes
            print(f"Eval: {avg_steps} avg steps, {avg_reward} avg reward")

            if args.track:
                wandb.log({"eval_cumulative_reward": avg_reward, "eval_episode_steps": avg_steps}, step=global_step, commit=False)

            if avg_reward > best_eval_reward:
                print("NEW BEST EVAL PERFORMANCE:", avg_reward)
                best_eval_reward = avg_reward
                save_model(actor, args.best_model_save_path)
            save_model(actor, args.last_model_save_path)

        if args.track:
            wandb.log({}, step=global_step, commit=False)

    env.close()
    env_eval.close()
