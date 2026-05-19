import numpy as np
import torch
from trajectory.utils import discretization
from trajectory.utils.arrays import to_torch
from .io import get_dataset, resolve_dataset_path

def segment(observations, terminals, max_path_length):
    """
        segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        if term.squeeze():
            trajectories.append([])

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros((n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype)
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=np.bool_)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i,:path_length] = traj
        early_termination[i,path_length:] = 1

    return trajectories_pad, early_termination, path_lengths

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env, sequence_length=250, step=10, discount=0.99, max_path_length=500, penalty=None, device='cuda:0', dataset_path=None):
        print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        self.device = device

        print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
        dataset = self._load_dataset(env, dataset_path)
        print('✓')

        # preprocess_fn = dataset_preprocess_functions.get(env.name)
        # if preprocess_fn:
        #     print(f'[ datasets/sequence ] Modifying environment')
        #     dataset = preprocess_fn(dataset)
        ##

        # observations = dataset['observations']
        # actions = dataset['actions']
        next_observations = dataset['next_observations'].astype(np.float32)
        # rewards = dataset['rewards']
        # terminals = dataset['terminals']
        observations = dataset['observations']
        observations = observations.astype(np.float32)

        actions = dataset['actions'].astype(np.float32)
        if actions.ndim == 1:
            actions = actions[:, np.newaxis]
        rewards = dataset['rewards'].astype(np.float32).reshape(-1, 1)
        terminals = dataset['terminals'].reshape(-1, 1)
        #realterminals = dataset['realterminals']

        self.observations_raw = observations
        self.actions_raw = actions
        self.next_observations_raw = next_observations
        self.joined_raw = np.concatenate([observations, actions], axis=-1)
        self.rewards_raw = rewards
        self.terminals_raw = terminals

        ## terminal penalty
        # if penalty is not None:
        #     terminal_mask = realterminals.squeeze()
        #     self.rewards_raw[terminal_mask] = penalty

        ## segment
        print(f'[ datasets/sequence ] Segmenting...', end=' ', flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(self.joined_raw, terminals, max_path_length)
        self.rewards_segmented, *_ = segment(self.rewards_raw, terminals, max_path_length)
        print('✓')

        self.discount = discount
        self.discounts = (discount ** np.arange(self.max_path_length))[:,None]

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:,t+1:] * self.discounts[:-t-1]).sum(axis=1)
            self.values_segmented[:,t] = V

        ## add (r, V) to `joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]
        self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
        self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)

        ## get valid indices
        indices = []
        for path_ind, length in enumerate(self.path_lengths):
            end = length - 1
            for i in range(end):
                indices.append((path_ind, i, i+sequence_length))

        self.indices = np.array(indices)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.joined_segmented = np.concatenate([
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length-1, joined_dim)),
        ], axis=1)
        self.termination_flags = np.concatenate([
            self.termination_flags,
            np.ones((n_trajectories, sequence_length-1), dtype=np.bool_),
        ], axis=1)

    def _load_dataset(self, env, dataset_path):
        resolved_dataset_path = resolve_dataset_path(env, dataset_path)
        if resolved_dataset_path is not None:
            return get_dataset(str(resolved_dataset_path))

        from .d4rl import load_environment, qlearning_dataset_with_timeouts
        from .preprocessing import dataset_preprocess_functions

        env = load_environment(env)
        dataset = qlearning_dataset_with_timeouts(env, terminate_on_end=True)
        preprocess_fn = dataset_preprocess_functions.get(env.name)
        if preprocess_fn:
            print(f'[ datasets/sequence ] Modifying environment')
            dataset = preprocess_fn(dataset)
        return dataset

    def __len__(self):
        return len(self.indices)


class DiscretizedDataset(SequenceDataset):

    def __init__(self, *args, N=50, discretizer='QuantileDiscretizer', **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        discretizer_class = getattr(discretization, discretizer)
        self.discretizer = discretizer_class(self.joined_raw, N)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step]

        joined_discrete = self.discretizer.discretize(joined)

        ## replace with termination token if the sequence has ended
        assert (joined[terminations] == 0).all(), \
                f'Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}'
        joined_discrete[terminations] = self.N

        ## [ (sequence_length / skip) x observation_dim]
        joined_discrete = to_torch(joined_discrete, device='cpu', dtype=torch.long).contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0

        ## flatten everything
        joined_discrete = joined_discrete.view(-1)
        mask = mask.view(-1)

        X = joined_discrete[:-1]
        Y = joined_discrete[1:]
        mask = mask[:-1]

        return X, Y, mask
class GoalDataset(DiscretizedDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        X, Y, mask = super().__getitem__(idx)

        ## get path length for looking up the last transition in the trajcetory
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]

        ## the goal is the first `observation_dim` dimensions of the last transition
        goal = self.joined_segmented[path_ind, path_length-1, :self.observation_dim]
        goal_discrete = self.discretizer.discretize(goal, subslice=(0, self.observation_dim))
        goal_discrete = to_torch(goal_discrete, device='cpu', dtype=torch.long).contiguous().view(-1)

        return X, goal_discrete, Y, mask
