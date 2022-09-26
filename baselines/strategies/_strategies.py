from copy import deepcopy
import numpy as np
from environment import State


def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items():
        if key in ('observation', 'static_info'):
            continue

        if key == 'capacity':
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res


def _greedy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[:] = True
    return _filter_instance(observation, mask)


def _lazy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    return _filter_instance(observation, mask)


def _random(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.5, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)


def _supervised(observation: State, rng: np.random.Generator, net):
    from baselines.supervised.transform import transform_one
    mask = np.copy(observation['must_dispatch'])
    mask = mask | net(transform_one(observation)).argmax(-1).bool().numpy()
    mask[0] = True
    return _filter_instance(observation, mask)


def _dqn(observation: State, rng: np.random.Generator, net):
    import torch
    from baselines.dqn.utils import get_request_features
    actions = []
    epoch_instance = observation
    observation, static_info = epoch_instance.pop('observation'), epoch_instance.pop('static_info')
    request_features, global_features = get_request_features(observation, static_info, net.k_nearest)
    all_features = torch.cat((request_features, global_features[None, :].repeat(request_features.shape[0], 1)), -1)
    actions = net(all_features).argmax(-1).detach().cpu().tolist()
    mask = epoch_instance['must_dispatch'] | (np.array(actions) == 0)
    mask[0] = True  # Depot always included in scheduling
    return _filter_instance(epoch_instance, mask)


def _somewhat_greedy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    greedy_ratio = (observation["observation"]["current_epoch"]+1) /  observation["static_info"]["num_epochs"]
    number_of_pending_request = (observation["is_depot"].shape[0]-1)
    n_request_to_dispatch = int(greedy_ratio*number_of_pending_request)
    # sorted_idx = np.argsort(observation["time_windows"][:,0])[::-1]
    sorted_idx_decrease_distance_to_depot = np.argsort(observation["duration_matrix"][0])[::-1]
    sorted_idx_decrease_tw_closeness = np.argsort(observation["time_windows"][:,0])[::-1]
    sorted_idx_decrease_demands = np.argsort(observation["demands"])[::-1]

    from copy import deepcopy
    request_difficulty = deepcopy(observation["request_idx"])
    for idx in range(request_difficulty.shape[0]):
        request_difficulty[idx] = np.where(sorted_idx_decrease_distance_to_depot==idx)[0][0] + (1/10)*np.where(sorted_idx_decrease_tw_closeness==idx)[0][0] + (1/10)*np.where(sorted_idx_decrease_demands==idx)[0][0]
    sorted_idx = np.argsort(request_difficulty)

    mask[sorted_idx[:n_request_to_dispatch]] = True
    mask[0] = True
    return _filter_instance(observation, mask)



STRATEGIES = dict(
    greedy=_greedy,
    lazy=_lazy,
    random=_random,
    supervised=_supervised,
    somewhat_greedy = _somewhat_greedy,
    dqn=_dqn
)
