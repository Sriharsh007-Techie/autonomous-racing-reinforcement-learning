import gymnasium as gym
import pickle
import numpy as np

from CarEnv.Configs import RACING_FAST
from CarEnv.Track.Generator import make_full_environment
from agent_interface import Agent


def save_model(model: Agent, save_path: str):
    """
    saves the provided actor to a file according to the given path
    :param model: the model to save
    :param save_path: where to save the actor to
    """
    model.to("cpu")  # for some models this might fail, in that case remove the line, but make sure the model is on cpu
    with open(save_path, 'wb') as filehandler:
        pickle.dump(model, filehandler)
    print(f"Saving actor as {save_path}")


def load_model(load_path: str):
    """
    loads and returns a previously saved actor
    :param load_path: where to load the actor from
    :return: the loaded model
    """

    with open(load_path, 'rb') as filehandler:
        model = pickle.load(filehandler)
    print(f"Loading actor from {load_path}")
    return model


def generate_tracks(num_tracks: int, save_path: str):
    """
    Generates num_tracks many random racing tracks and saves them as a list of edicts containing the track information
    to the provided save_path.
    Note that you can pass a previously generated track 'example_track' to the environments reset function, in order
    to use this instead of a newly generated random track, as follows:
    reset(seed=True, options={'predefined_track': track})
    :param num_tracks: how many tracks to generate
    :param save_path: where to save the generated tracks to
    """

    tracks = []

    for i in range(num_tracks):
        track = make_full_environment(width=RACING_FAST['problem']['track_width'],
                                      extends=(RACING_FAST['problem']['extend'], RACING_FAST['problem']['extend']),
                                      cone_width=RACING_FAST['problem']['cone_width'],
                                      rng=np.random.default_rng())
        tracks += [track]

    with open(save_path, 'wb') as filehandler:
        pickle.dump(tracks, filehandler)
    print(f"Saving tracks at {save_path}")


def load_tracks(load_path: str):
    """
    loads and returns tracks from the given load_path
    :param load_path: path to the track file
    :return: the loaded tracks
    """

    with open(load_path, 'rb') as filehandler:
        tracks = pickle.load(filehandler)
    return tracks


def create_env(seed: int, render_env: bool = False, limit_speed_factor=None, render_width: int = 1280):
    """
    creates a racing environment
    (hint: it might make sense to use rendering for evaluation, but be aware that rendering slows the environment down)
    :param seed: (random) seed for the environment creation
    :param render_env: Can be set to True to render the environment steps with Pygame
    :param limit_speed_factor: Factor by which the rendering is at most sped up (if None: render as fast as possible)
    :param render_width: width of the Pygame rendering (in pixels)
    :return: the racing environment
    """

    env = gym.make('CarEnv:gym_envs/CarEnv-v1', config=RACING_FAST, render_env=render_env,
                   limit_speed_factor=limit_speed_factor, render_width=render_width)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return gym.wrappers.RecordEpisodeStatistics(env, deque_size=6000)
