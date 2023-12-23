"""
# example
import torch
import os
import random
import time
from dataclasses import dataclass
from gymnasium.spaces import Box
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn, optim
import torch.nn.functional as F

from agent_interface import convert_action, convert_obs, Agent
from util import *


@dataclass
class Args:
    exp_name: str = "sample_solution_ddpg"
    # the name of this experiment

    seed: int = 42
    # seed of the experiment

    torch_deterministic: bool = True
    # if toggled, `torch.backends.cudnn.deterministic=False`

    track: bool = True
    # if toggled, this experiment will be tracked with Weights and Biases

    wandb_project_name: str = "RLLBC_BPA3"
    # the wandb's project name

    wandb_entity: str = None
    # the entity (team) of wandb's project

    save_model: bool = True
    # whether to save model into the `runs/{run_name}` folder

    env_id: str = "Racing-Env"
    # the environment id

    total_timesteps: int = 3000000
    # total timesteps of the experiments

    learning_rate: float = 1e-5
    # the learning rate of the optimizer

    buffer_size: int = int(5e5)
    # the replay memory buffer size

    gamma: float = 0.99
    # the discount factor gamma

    tau: float = 0.005
    # target smoothing coefficient (default: 0.005)

    batch_size: int = 512
    # the batch size of sample from the reply memory

    exploration_noise: float = 0.1
    # the scale of exploration noise

    learning_starts: int = 1e4
    # timestep to start learning

    policy_frequency: int = 2
    # the frequency of training policy (delayed)

    noise_clip: float = 0.5
    # noise clip parameter of the Target Policy Smoothing Regularization

    eval_freq: int = 50000
    # frequency of evaluation

    render_eval: bool = True
    # whether to render the evaluation episodes

    num_eval_episodes: int = 50
    # how many episodes to run at each evaluation time

    best_model_save_path: str = os.path.join("path_to_some_directory", exp_name + "_best")
    # where to save the model with the best evaluation performance

    last_model_save_path: str = os.path.join("path_to_some_directory", exp_name + "_last")
    # where to save the last model state

    track_path: str = "path_to_save_tracks_to"
    # where to save the generated tracks


def make_env(seed: int, render_env: bool = False, limit_speed_factor=None, render_width: int = 1280):
    def thunk():
        env = create_env(seed, render_env, limit_speed_factor, render_width)
        return env
    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        action_shape = envs.action_space.shape
        obs_shape = 415  # set manually as we change the shape in convert_obs()

        self.fc1 = nn.Linear(obs_shape + np.prod(action_shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    args = Args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # first we generate and save some tracks to use during evaluation (instead of generating new tracks on every reset)
    # -> leads to more stable evaluation performances that can be interpreted more reliably
    track_path = "path_to_tracks"
    generate_tracks(num_tracks=args.num_eval_episodes, save_path=track_path)
    eval_tracks = load_tracks(track_path)

    # it can be very helpful to track the training using weights and biases
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([(make_env(seed=args.seed + i)) for i in range(1)])
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    envs_eval = create_env(args.seed, render_env=True, limit_speed_factor=None, render_width=1280)
    assert isinstance(envs_eval.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Agent(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Agent(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # define Box that fits the shape of the observations obtained from convert_obs()
    # with this we can initialize a Replay Buffer for our converted observations
    new_shape = (envs.single_observation_space.shape[0] - 1,)
    new_box = Box(low=envs.single_observation_space.low[:new_shape[0]], high=envs.single_observation_space.high[:new_shape[0]])
    envs.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        new_box,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = convert_obs(obs).to(device)

    episode_step = 0
    best_eval_reward = -np.inf

    for global_step in range(args.total_timesteps):

        episode_step += 1

        # action logic
        if global_step < args.learning_starts:
            actions = envs.single_action_space.sample()
        else:
            with torch.no_grad():
                actions = convert_action(actor(obs))
                actions += torch.normal(0, actor.action_scale[0] * args.exploration_noise).cpu().numpy()
                actions = actions.clip(envs.single_action_space.low, envs.single_action_space.high)

        # execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions[np.newaxis, :])
        next_obs = convert_obs(next_obs).to(device)

        # record rewards for plotting purposes and logging
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                print(f"global_step={global_step}, environment {i}: episodic_return={info['episode']['r'][i]}")
                wandb.log({'train_reward': info['episode']['r'][i], 'train_ep_steps': episode_step, 'step': global_step})
                episode_step = 0

        # save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.cpu().numpy().copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs = convert_obs(infos["final_observation"][idx])
        rb.add(obs.cpu(), real_next_obs, actions, rewards, terminations, infos)

        # CRUCIAL step easy to overlook
        obs = next_obs

        # training logic
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                wandb.log({'actor_loss': actor_loss.item(), 'step': global_step})
            wandb.log({'qf1_loss': qf1_loss.item(), 'step': global_step})

        # evaluation
        eval_episodes_performed = 0
        if global_step > args.learning_starts and global_step % args.eval_freq == 0:  # do an eval episode

            total_reward = 0
            total_eval_steps = 0

            for i in range(args.num_eval_episodes):

                print(f"Eval episode {i+1} started...")

                done = False
                truncation_eval = False
                eval_step = 0

                if i >= len(eval_tracks):
                    print("No more tracks left!")
                    break

                obs_eval, _ = envs_eval.reset(seed=True, options={'predefined_track': eval_tracks[i]})
                obs_eval = convert_obs(obs_eval).to(device)

                while not done and not truncation_eval:
                    eval_step += 1
                    with torch.no_grad():
                        action = convert_action(actor.get_action(obs_eval))
                    obs_eval, reward_eval, done, truncation_eval, info_eval = envs_eval.step(action)
                    obs_eval = convert_obs(obs_eval).to(device)
                    total_reward += reward_eval

                total_eval_steps += eval_step
                eval_episodes_performed += 1

            # logging evaluation performance
            avg_eval_reward = total_reward / eval_episodes_performed
            avg_eval_steps = total_eval_steps / eval_episodes_performed
            print(f"Evaluation result: {avg_eval_steps} avg. steps, avg. reward: {avg_eval_reward}")
            wandb.log({'eval_reward': avg_eval_reward, 'step': global_step})
            wandb.log({'eval_ep_steps': avg_eval_steps, 'step': global_step})

            # save model state
            if avg_eval_reward > best_eval_reward:
                print("NEW BEST EVALUATION PERFORMANCE")
                best_eval_reward = avg_eval_reward
                save_model(actor, args.best_model_save_path)
            save_model(actor, args.last_model_save_path)

    envs.close()
    envs_eval.close()
"""

# ToDo: Your training code here...
raise NotImplementedError
