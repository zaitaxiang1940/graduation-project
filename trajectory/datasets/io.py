import os
import pickle

import numpy as np

from trajectory.utils.paths import DATA_DIR, resolve_runtime_path


def resolve_dataset_path(env, dataset_path=None):
    candidate = dataset_path or os.environ.get('TRAJECTORY_DATASET_PATH')
    if candidate is None and isinstance(env, str) and env.endswith('.pkl'):
        candidate = env
    if candidate is None:
        default_candidate = DATA_DIR / 'training_data.pkl'
        if default_candidate.exists():
            candidate = default_candidate
    if candidate is None:
        return None
    return resolve_runtime_path(candidate, search_dirs=[DATA_DIR])


def get_dataset(file_path):
    with open(file_path, 'rb') as f:
        collected_data = pickle.load(f)
        timeout_key = 'timeouts' if 'timeouts' in collected_data else 'real_done'
        flattened_data = {
            'actions': np.concatenate(collected_data['actions']),
            'observations': np.concatenate(collected_data['observations']),
            'next_observations': np.concatenate(collected_data['next_observations']),
            'rewards': np.concatenate(collected_data['rewards']),
            'terminals': np.concatenate(collected_data['terminals']),
            'timeouts': np.concatenate(collected_data[timeout_key]),
        }
    return flattened_data
