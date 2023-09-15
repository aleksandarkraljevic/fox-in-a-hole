import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from fox_in_a_hole import *

def exponential_anneal(t, start, final, decay_constant):
    ''' 
    Exponential annealing scheduler for epsilon-greedy policy.
    param t:        current timestep
    param start:    initial value
    param final:    value after percentage*T steps
    '''
    return final + (start - final) * np.exp(-decay_constant*t)


def boltzmann_exploration(actions, temperature):
    '''
    Boltzmann exploration policy.
    param actions:      vector with possible actions
    param temperature:  exploration parameter
    return:             vector with probabilities for choosing each option
    '''
    # print(f'bolzmann exploration of {actions}')  # can remove this line once everything works
    actions = actions[0] / temperature  # scale by temperature
    a = actions - max(actions)  # substract maximum to prevent overflow of softmax
    return np.exp(a)/np.sum(np.exp(a))

def plot(episodes, episode_lengths, show, savename):
    plt.figure()
    plt.plot(episodes, episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Guesses')
    if savename != False:
        plt.savefig(savename)
    if show:
        plt.show()

def test_performance(model_name, n_samples, n_holes, print_strategy):
    model = tf.keras.models.load_model(model_name)
    env = FoxInAHole(n_holes, 2 * n_holes)
    observation = [0] * (2 * n_holes)
    fox, done = env.reset()
    won, lost = done
    current_episode_length = 0
    episode_lengths = []
    for sample in range(n_samples):
        while (not won) and (not lost):
            current_episode_length += 1
            predicted_q_values = model.predict(np.asarray(observation).reshape(1, 2 * n_holes), verbose=0)
            action = np.argmax(predicted_q_values) + 1
            reward, done = env.guess(action, current_episode_length)
            won, lost = done
            new_observation = observation.copy()
            new_observation[current_episode_length - 1] = action
            observation = new_observation
            if won or lost:
                if print_strategy:
                    print(observation)
                episode_lengths.append(current_episode_length)
                current_episode_length = 0
                fox, done = env.reset()
                observation = [0] * (2 * n_holes)
                break
            fox = env.step()
    print('The average amount of guesses needed to finish the game is: ',np.mean(episode_lengths))
