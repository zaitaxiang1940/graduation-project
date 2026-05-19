import argparse
import itertools
import random

import numpy as np
import torch

from _settings import Settings
from trajectory.utils.paths import build_project_path


def do_task():
    if Settings.TASK == "ST":
        import sumo
        import st
        sumo.start_sumo()
        st.evaluate_st_and_dump_crash(Settings.NUM_EPISODES)
        sumo.exit_sumo()
    elif Settings.TASK == "TRAIN_DQN":
        import rainbow
        rainbow.train_rainbow_all_with_lr_drop(1e6)
    elif Settings.TASK == "TRAIN_DDPG":
        import ddpg
        ddpg.train_ddpg_all_with_lr_drop(1e6)
    elif Settings.TASK == "RESUME_DQN":
        import rainbow
        rainbow.RainbowAgent.resume_training(Settings.MODEL_NAME, 1e6)
    elif Settings.TASK == "RESUME_DDPG":
        import ddpg
        ddpg.DDPGAgent.resume_training(Settings.MODEL_NAME, 1e6)
    elif Settings.TASK == "EVALUATE_DQN":
        import rainbow
        eval_agent = rainbow.RainbowAgent.load(Settings.MODEL_NAME)
        eval_agent.evaluate(Settings.NUM_EPISODES)
    elif Settings.TASK == "EVALUATE_DDPG":
        import ddpg
        eval_agent = ddpg.DDPGAgent.load(Settings.MODEL_NAME)
        eval_agent.evaluate(Settings.NUM_EPISODES)
    elif Settings.TASK == "EVALUATE_COMBINED_DQN":
        import ddpg
        eval_agent = ddpg.DDPGAgent.load(Settings.MODEL_NAME)
        eval_agent.evaluate_combined(Settings.NUM_EPISODES)
    elif Settings.TASK == "EVALUATE_COMBINED_DDPG":
        import ddpg
        eval_agent = ddpg.DDPGAgent.load(Settings.MODEL_NAME)
        eval_agent.evaluate_combined(Settings.NUM_EPISODES)


def do_grid_search_st():
    search_grid = {
        "V_WEIGHT": [0.5, 1.0],
        "A_WEIGHT": [0.0, 10.0],
        "J_WEIGHT": [0.0, 10.0, 50.0],
        "D_WEIGHT": [0.0, 10.0, 100.0, 1000.0],
        "MIN_ALLOWED_DISTANCE": [5, 6],
        "CRASH_MIN_S": [10, 15, 20]
    }

    items = list(search_grid.items())
    keys = [x[0] for x in items]
    values = [x[1] for x in items]
    for value_tuple in itertools.product(*values):
        for i, key in enumerate(keys):
            setattr(Settings, key, value_tuple[i])
        do_task()


def do_grid_search_combined():
    search_grid = {
        "ROLLOUT_LENGTH": [3, 5, 10, 20],
        "ST_TEST_ROLLOUTS": [2, 5, 10],
        "TEST_ROLLOUT_STATE": [True, False]
    }

    items = list(search_grid.items())
    keys = [x[0] for x in items]
    values = [x[1] for x in items]
    for value_tuple in itertools.product(*values):
        for i, key in enumerate(keys):
            setattr(Settings, key, value_tuple[i])
        if not Settings.TEST_ROLLOUT_STATE and Settings.ST_TEST_ROLLOUTS != 2:
            continue
        if Settings.ROLLOUT_LENGTH == 1 and Settings.ST_TEST_ROLLOUTS != 2:
            continue
        if Settings.ST_TEST_ROLLOUTS > Settings.ROLLOUT_LENGTH:
            continue
        do_task()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs='?', default=str(build_project_path("configs", "combined_default_1.json")))
    args = parser.parse_args()
    config_file = args.config
    if config_file is not None:
        Settings.load_from_file(config_file)
    Settings.setup_logging()
    if Settings.TASK != "ST":
        import merge_gym
        merge_gym.register_environments()

    if Settings.SEED != "Random":
        np.random.seed(Settings.SEED)
        torch.manual_seed(Settings.SEED)
        random.seed(Settings.SEED)
        if Settings.CUDA:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    do_task()
