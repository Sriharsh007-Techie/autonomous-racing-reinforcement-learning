from CarEnv.Configs import RACING_FAST
from agent_interface import convert_obs, convert_action, Agent
from util import *
import torch


"""
Example evaluation script to test (and visualize) the performance of a previously saved agent on random tracks
"""


load_path_model = "path_to_agent"  # replace this by the actual path to your previously saved actor
num_episodes = 42
seed = 42
predef_tracks = False  # you could use generate_tracks() from util.py to save tracks and use them instead of random ones
load_path_tracks = "path_to_tracks"  # replace by a path if you want to load previously saved tracks


model = load_model(load_path_model)
env = gym.make('CarEnv:gym_envs/CarEnv-v1', config=RACING_FAST)

total_reward = 0
total_eval_steps = 0

print(f"Starting evaluation of actor saved in {load_path_model}.")
print(f"Performing {num_episodes} evaluation runs...")

if predef_tracks:
    tracks = load_tracks(load_path_tracks)
    num_episodes = min(num_episodes, len(tracks))

returns = [0]*num_episodes

for i in range(num_episodes):

    if tracks is not None:
        if i >= len(tracks):
            print("No more tracks left!")
            break
        obs, _ = env.reset(seed=seed+i, options={'predefined_track': tracks[i]})
    else:
        obs, _ = env.reset(seed=seed+i)

    done = False
    while not done:
        state = convert_obs(obs)
        with torch.no_grad():
            output = model.get_action(state)
        action = convert_action(output)

        obs, reward, terminated, truncated, info = env.step(action)
        returns[i] += reward
        done = terminated or truncated

env.close()

print(f"Mean return: {np.mean(returns)}")
print(f"Std. deviation: {np.std(returns)}")
