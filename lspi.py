import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer

import matplotlib.pyplot as plt


def compute_lspi_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma):
    # compute the next w given the data.
    N = encoded_states.shape[0]

    phi_s_a = np.zeros((encoded_states.shape[0], 3*encoded_states.shape[1] + 3))
    for i, a in enumerate(actions):
        phi_s_a[i, a*(encoded_states.shape[1]+1): (a+1)*encoded_states.shape[1] + a] = encoded_states[i]
        phi_s_a[i, (a+1)*encoded_states.shape[1] + a] = 1

    phi_s_a_next = np.zeros((encoded_next_states.shape[0], 3*encoded_next_states.shape[1] + 3))
    linear_policy_next = linear_policy.get_max_action(encoded_next_states)
    for i, a in enumerate(linear_policy_next):
        phi_s_a_next[i, a*(encoded_next_states.shape[1]+1): (a+1)*encoded_next_states.shape[1] + a] = encoded_next_states[i]
        phi_s_a_next[i, (a+1)*encoded_next_states.shape[1] + a] = 1

    A = (1/N) * phi_s_a.T @ (phi_s_a - gamma * phi_s_a_next)
    b = (1/N) * np.sum(np.expand_dims(rewards, axis=-1) * phi_s_a, axis=0)

    next_w = np.linalg.solve(A, b)

    return np.expand_dims(next_w, axis=-1)


def run_3_seeds_and_evaluate():
    success_rates_all_seeds = []
    for seed in [123, 456, 1995]:
        success_rates = []

        np.random.seed(seed)

        samples_to_collect = 100000
        # samples_to_collect = 150000
        # samples_to_collect = 10000
        number_of_kernels_per_dim = [12, 10]
        gamma = 0.999
        w_updates = 20
        evaluation_number_of_games = 50
        evaluation_max_steps_per_game = 500

        env = MountainCarWithResetEnv()
        # collect data
        states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
        # get data success rate
        data_success_rate = np.sum(rewards) / len(rewards)
        print(f'success rate {data_success_rate}')
        # standardize data
        data_transformer = DataTransformer()
        data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
        states = data_transformer.transform_states(states)
        next_states = data_transformer.transform_states(next_states)
        # process with radial basis functions
        feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        # encode all states:
        encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
        encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
        # set a new linear policy
        linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
        # but set the weights as random
        linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
        # start an object that evaluates the success rate over time
        evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
        for lspi_iteration in range(w_updates):
            print(f'starting lspi iteration {lspi_iteration}')

            new_w = compute_lspi_iteration(
                encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
            )
            norm_diff = linear_policy.set_w(new_w)

            success_rates.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))

            if norm_diff < 0.00001:
                break
        print('done lspi')

        success_rates_all_seeds.append(success_rates)

    # plot all 3 in one graph:
    for i in range(3):
        plt.plot(success_rates_all_seeds[i])

    plt.xlabel('iteration')
    plt.ylabel('success rate')
    plt.show()

    # plot average

def run_different_num_of_samples():
    samples = [5000, 20000, 50000, 100000, 200000, 500000, 1_000_000]
    res = []
    for num_samples in samples:
        samples_to_collect = num_samples
        # samples_to_collect = 150000
        # samples_to_collect = 10000
        number_of_kernels_per_dim = [12, 10]
        gamma = 0.999
        w_updates = 20
        evaluation_number_of_games = 50
        evaluation_max_steps_per_game = 1000

        np.random.seed(123)
        np.random.seed(42)

        env = MountainCarWithResetEnv()
        # collect data
        states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
        # get data success rate
        data_success_rate = np.sum(rewards) / len(rewards)
        print(f'success rate {data_success_rate}')
        # standardize data
        data_transformer = DataTransformer()
        data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
        states = data_transformer.transform_states(states)
        next_states = data_transformer.transform_states(next_states)
        # process with radial basis functions
        feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        # encode all states:
        encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
        encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
        # set a new linear policy
        linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
        # but set the weights as random
        linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
        # start an object that evaluates the success rate over time
        evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
        for lspi_iteration in range(w_updates):
            print(f'starting lspi iteration {lspi_iteration}')

            new_w = compute_lspi_iteration(
                encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
            )
            norm_diff = linear_policy.set_w(new_w)
            if norm_diff < 0.00001:
                break
        print('done lspi')
        res.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))

    plt.plot(samples, res)
    plt.xlabel('number of samples')
    plt.ylabel('success rate')
    plt.xscale('log')
    plt.show()
    pass

if __name__ == '__main__':
    run_3_seeds_and_evaluate()
    run_different_num_of_samples()