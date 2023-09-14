import gym
import numpy as np
import pygame
import tensorflow as tf
from collections import deque
from tqdm import tqdm
import time
import random

from fox_in_a_hole import *
from helper import *

def initialize_model(n_holes, learning_rate):
    '''
    Build the model. It is a simple Neural Network consisting of 3 densely connected layers with Relu activation functions.
    The only argument is the learning rate.
    '''
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(24, activation='relu', input_shape=(2*n_holes,), kernel_initializer=tf.keras.initializers.GlorotUniform()),
      tf.keras.layers.Dense(48, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform()),
      tf.keras.layers.Dense(n_holes, activation='linear', kernel_initializer=tf.keras.initializers.GlorotUniform())
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return model


def update_model(base_model, target_network):
    '''
    Copies weights from base model to target network.
    param base_model:       tf base model
    param target_network:   tf target network
    '''
    for layer_TN, layer_BM in zip(target_network.layers, base_model.layers):
        layer_TN.set_weights(layer_BM.get_weights())


def train(base_model, target_network, replay_buffer, activate_ER, activate_TN, learning_rate, n_holes = 5):
    '''
    Trains the model using the DQN algorithm. 
    The Replay Experience buffer (if enabled) is used to indicate which states we want to train the model on. 
    Otherwise, we use the last state observed in the list.
    Then, it predicts the new Q-values with the use of the Target Network (if enabled).
    Finally it fits the model (using a batch size if Replay Experience buffer is enabled).
    param base_model:       the constructed Model
    param target_network:   Target Network
    param replay_buffer:    Experience Replay buffer for storing states.
    param activate_ER:      True of False whether an Experience Replay Buffer is used 
    param activate_TN:      True of False whether a Target Network is used 
    param learning_rate:    learning rate hyperparameter
    '''
    last_element = -1 # index for the last element of the buffer for if we update without batches
    won, lost = replay_buffer[last_element][4], replay_buffer[last_element][5]

    if not activate_ER:                    # for the baseline: just take the last element
        sample_list = [last_element]    
    else:                                  # for the ER: check the conditions and then take a sample
        min_size_buffer = 1_000
        batch_size = 128

        if len(replay_buffer) < min_size_buffer:
            return
        
        sample_list = random.sample(range(0, len(replay_buffer)), batch_size)

    observation_list = list()
    new_observation_list = list()
    action_list = list()
    reward_list = list()
    won_list = list()
    lost_list = list()
    for element in sample_list:
        observation_list.append(replay_buffer[element][0])
        new_observation_list.append(replay_buffer[element][3])
        action_list.append(replay_buffer[element][1])
        reward_list.append(replay_buffer[element][2])
        won_list.append(replay_buffer[element][4])
        lost_list.append(replay_buffer[element][5])

    predicted_q_values = base_model.predict(np.array(observation_list),verbose=0)
    if activate_TN:
        new_predicted_q_values = target_network.predict(np.array(new_observation_list),verbose=0)
    else:
        new_predicted_q_values = base_model.predict(np.array(new_observation_list),verbose=0)

    q_bellman_list = list()
    for i in range(len(observation_list)):
        if (not won_list[i]) and (not lost_list[i]):
            q_bellman = predicted_q_values[i] - learning_rate * (predicted_q_values[i] - reward_list[i] - gamma * max(new_predicted_q_values[i]))
        else:
            q_bellman = predicted_q_values[i] - learning_rate * (predicted_q_values[i] - reward_list[i])
        # here we have to replace the q values of the actions that we didn't choose to take back with their previous values, such that only the taken state-action is updated
        possible_actions = list(range(n_holes))
        possible_actions.pop(action_list[i]-1)
        for j in possible_actions:
            q_bellman[j] = predicted_q_values[i][j]
        q_bellman_list.append(q_bellman)
    
    if activate_ER:
        base_model.fit(x=np.array(observation_list), y=np.array(q_bellman_list), batch_size=batch_size, verbose=0)
    else:
        base_model.fit(x=np.array(observation_list), y=np.array(q_bellman_list), verbose=0)


def main(base_model, target_network, num_episodes, initial_exploration, final_exploration, learning_rate, decay_constant, temperature, activate_TN, activate_ER, exploration_strategy='anneal_epsilon_greedy', n_holes = 5):
    '''
    For all the episodes, the agent selects an action (based on the given policy) and then trains the model.
    Experience Replacy and Target Network are used if they were specified when this function was called.
    same parameters as the train function: base_model, target_network, learning_rate, activate_TN, activate_ER
    param num_episodes:             integet number specifying the number of episodes
    param initial_exploration:      upper limit of epsilon value for annealing epsilon greedy
    param final_exploration:        lower limit of epsilon value for annealing epsilon greedy
    param decay_constant:           decreasing value for annealing epsilon greedy
    param temperature:              key parameter of boltzmann's policy
    param exploration_strategy:     by default is set to 'anneal_epsilon_greedy' but 'boltzmann' is also a valid option     
    '''
    env = FoxInAHole(n_holes)

    episode_lengths = []
    replay_buffer = deque(maxlen=1000)
    current_episode_length = 0

    if activate_TN:     # start by copying over the weights from TN to base model to ensure they are identical
        update_model(base_model=base_model, target_network=target_network)
        steps_TN = 0

    observation, fox = env.reset()

    for episode in tqdm(range(num_episodes)):
        won, lost = False, False # won and lost represent whether the game has been won or lost yet

        if exploration_strategy == 'anneal_epsilon_greedy':
            # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
            exploration_parameter = exponential_anneal(episode, initial_exploration, final_exploration, decay_constant)
            epsilon = exploration_parameter  # temporary while only using egreedy

        while (not won) and (not lost):
            current_episode_length += 1

            # let the main model predict the Q values based on the observation of the environment state
            # these are Q(S_t)
            predicted_q_values = base_model.predict([observation],verbose=0)

            # choose an action
            if exploration_strategy == 'anneal_epsilon_greedy':
                if np.random.random() < epsilon:    # exploration
                    action = np.random.randint(1, n_holes+1)
                else:
                    action = np.argmax(predicted_q_values) + 1  # exploitation: take action with highest associated Q value
            elif exploration_strategy == 'boltzmann':
                probabilities = np.cumsum(boltzmann_exploration(predicted_q_values, temperature))
                random_number = np.random.random()
                action = np.argmax(random_number < probabilities) + 1  # numpy argmax takes first True value

            # for testing:
            #print(f'predicted Q values {predicted_q_values}')
            #print(f'Chosen action: {action}')

            new_observation, reward, done = env.guess(action, current_episode_length)
            won, lost = done
            print(observation) ##########
            replay_buffer.append([observation, action, reward, new_observation, won, lost])

            if activate_TN:
                steps_TN += 1
                if current_episode_length % 4 == 0 or won or lost:
                    train(base_model=base_model, target_network=target_network, replay_buffer=replay_buffer, activate_ER=activate_ER, activate_TN=activate_TN, learning_rate=learning_rate, n_holes=n_holes)
            else:
                train(base_model=base_model, target_network=target_network, replay_buffer=replay_buffer, activate_ER=activate_ER, activate_TN=activate_TN, learning_rate=learning_rate, n_holes=n_holes)

            # roll over
            observation = new_observation

            if won or lost:
                episode_lengths.append(current_episode_length)
                current_episode_length = 0
                observation, fox = env.reset()

                if activate_TN:
                    if steps_TN >= update_freq_TN:
                        update_model(base_model=base_model, target_network=target_network)  # copy over the weights
                        steps_TN = 0
                break

            fox = env.step()

    return episode_lengths


if __name__ == '__main__':
    # game parameters
    n_holes = 5
    # Hyperparameters of the algorithm and other parameters of the program
    learning_rate = 0.01
    gamma = 1  # discount factor
    initial_epsilon = 1  # 100%
    final_epsilon = 0.01  # 1%
    num_episodes = 10
    decay_constant = 0.1  # the amount with which the exploration parameter changes after each episode
    temperature = 0.1
    activate_ER = False
    activate_TN = False
    exploration_strategy = 'anneal_epsilon_greedy'

    start = time.time()

    base_model = initialize_model(n_holes=n_holes, learning_rate=learning_rate)
    target_network = initialize_model(n_holes=n_holes, learning_rate=learning_rate)
    if activate_TN:
        update_freq_TN = 10

    episode_lengths = main(base_model=base_model, target_network=target_network, num_episodes=num_episodes, initial_exploration=initial_epsilon, final_exploration=final_epsilon, learning_rate=learning_rate, decay_constant=decay_constant, temperature=temperature, activate_TN=activate_TN, activate_ER=activate_ER, exploration_strategy='anneal_epsilon_greedy', n_holes=n_holes)
    print('episode lengths = ', episode_lengths)

    end = time.time()
    print('Total time: {} seconds (number of episodes: {})'.format(end-start, num_episodes))