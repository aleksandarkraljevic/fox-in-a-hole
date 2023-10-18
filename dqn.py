from collections import deque
from tqdm import tqdm
import time
import random

from helper import *
from classic_NN import *


class DQN():
    def __init__(self, savename, base_model, target_network, n_holes, memory_size, learning_rate, gamma, num_episodes, tau, initial_exploration, final_exploration, decay_constant, temperature, batch_size, min_size_buffer, max_size_buffer, exploration_strategy):
        self.savename = savename
        self.n_holes = n_holes
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.base_model = base_model
        self.target_network = target_network
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.tau = tau
        self.initial_exploration = initial_exploration
        self.final_exploration = final_exploration
        self.decay_constant = decay_constant
        self.temperature = temperature
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.max_size_buffer = max_size_buffer
        self.exploration_strategy = exploration_strategy

    def update_model(self):
        '''
        Copies weights from base model to target network via a soft-update rule.
        param base_model:       tf base model
        param target_network:   tf target network
        '''
        for layer_TN, layer_BM in zip(self.target_network.layers, self.base_model.layers):
            # soft update
            TN_weights = layer_TN.get_weights()
            BM_weights = layer_BM.get_weights()
            new_weights = []
            new_weights.append((1-self.tau) * TN_weights[0] + self.tau * BM_weights[0])
            new_weights.append((1 - self.tau) * TN_weights[1] + self.tau * BM_weights[1])
            layer_TN.set_weights(new_weights)
            # hard update
            # layer_TN.set_weights(layer_BM.get_weights())

    def save_data(self, rewards, episode_lengths):
        data = {'n_holes': self.n_holes, 'memory_size': self.memory_size, 'batch_size': self.batch_size, 'min_size_buffer': self.min_size_buffer, 'max_size_buffer': self.max_size_buffer, 'rewards': rewards, 'episode_lengths': episode_lengths}
        np.save('data/' + self.savename + '.npy', data)
        self.target_network.save('models/' + self.savename + '.keras')

    def custom_predict(self, observations, model, batch_size):
        predicted_q_values = []
        for observation in observations:
            predicted_q_values.append(model(np.asarray(observation).reshape(1,self.memory_size)))
        return np.reshape(predicted_q_values, (batch_size,self.n_holes))

    def train(self, replay_buffer):
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

        sample_list = random.sample(range(0, len(replay_buffer)), self.batch_size)

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

        predicted_q_values = self.custom_predict(observation_list, self.base_model, self.batch_size)

        new_predicted_q_values = self.custom_predict(new_observation_list, self.target_network, self.batch_size)

        q_bellman_list = list()
        for i in range(len(observation_list)):
            if (not won_list[i]) and (not lost_list[i]):
                q_bellman = predicted_q_values[i] + self.learning_rate * (reward_list[i] + self.gamma * max(new_predicted_q_values[i]) - predicted_q_values[i])
            else:
                q_bellman = predicted_q_values[i] + self.learning_rate * (reward_list[i] - predicted_q_values[i])
            # here we have to replace the q values of the actions that we didn't choose to take back with their previous values, such that only the taken state-action is updated
            possible_action_indices = list(range(self.n_holes))
            possible_action_indices.pop(action_list[i]-1)
            for j in possible_action_indices:
                q_bellman[j] = predicted_q_values[i][j]
            q_bellman_list.append(q_bellman)

        self.base_model.fit(x=np.asarray(observation_list), y=np.asarray(q_bellman_list), batch_size=self.batch_size, verbose=0)

    def main(self):
        '''
        For all the episodes, the agent selects an action (based on the given policy) and then trains the model.
        Experience Replacy and Target Network are used if they were specified when this function was called.
        same parameters as the train function: base_model, target_network, learning_rate, activate_TN, activate_ER
        param num_episodes:             integet number specifying the number of episodes
        param initial_exploration:      upper limit of epsilon value for annealing epsilon greedy
        param final_exploration:        lower limit of epsilon value for annealing epsilon greedy
        param decay_constant:           decreasing value for annealing epsilon greedy
        param temperature:              key parameter of boltzmann's policy
        param exploration_strategy:     by default is set to 'egreedy' but 'boltzmann' is also a valid option
        '''

        env = FoxInAHole(self.n_holes, self.memory_size)

        episode_lengths = []
        rewards = []
        replay_buffer = deque(maxlen=self.max_size_buffer)
        current_episode_length = 0
        total_steps = 0
        observation = [0] * self.memory_size # The memory of actions that have been taken is the observation
        possible_actions = list(range(self.n_holes))
        possible_actions = [x+1 for x in possible_actions]

        self.update_model() # start by copying over the weights from TN to base model to ensure they are identical

        done = env.reset()

        for episode in tqdm(range(self.num_episodes)):
            won, lost = done # won and lost represent whether the game has been won or lost yet

            if self.exploration_strategy == 'egreedy':
                # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
                exploration_parameter = exponential_anneal(episode, self.initial_exploration, self.final_exploration, self.decay_constant)

            while (not won) and (not lost):
                current_episode_length += 1
                total_steps += 1

                # let the main model predict the Q values based on the observation of the environment state
                # these are Q(S_t)
                predicted_q_values = self.base_model(np.asarray(observation).reshape(1,self.memory_size))

                # choose an action
                if self.exploration_strategy == 'egreedy':
                    if np.random.random() < exploration_parameter:    # exploration
                        action = np.random.randint(1, self.n_holes+1)
                    else:
                        action = np.argmax(predicted_q_values) + 1  # exploitation: take action with highest associated Q value
                elif self.exploration_strategy == 'boltzmann':
                    probabilities = boltzmann_exploration(predicted_q_values, self.temperature)
                    action = np.random.choice(possible_actions, p=probabilities)

                reward, done = env.guess(action, current_episode_length)
                won, lost = done
                new_observation = observation.copy()
                new_observation[current_episode_length-1] = action
                replay_buffer.append([observation, action, reward, new_observation, won, lost])

                if (total_steps % 10 == 0) and (len(replay_buffer) > self.min_size_buffer): # the model is trained after every game, as long as the replay buffer is filled up enough
                    self.train(replay_buffer)
                    self.update_model()  # copy over the weights only after a certain amount of training steps have been taken

                # roll over
                observation = new_observation
                env.step()

            episode_lengths.append(current_episode_length)
            rewards.append(reward)
            current_episode_length = 0
            done = env.reset()
            observation = [0] * self.memory_size

        if self.savename != False:
            self.save_data(rewards, episode_lengths)

def main():
    # name that will be used to save both the model and all its data with
    savename = 'test'
    # game parameters
    n_holes = 5
    memory_size = 2*n_holes
    # model parameters
    n_nodes = 12
    # Hyperparameters of the algorithm and other parameters of the program
    learning_rate = 0.01
    gamma = 1  # discount factor
    tau = 0.1 # TN soft-update speed parameter, tau is the ratio of the TN that gets copied over at each training step
    initial_exploration = 1  # 100%
    final_exploration = 0.01  # 1%
    num_episodes = 10000
    decay_constant = 0.0001  # the amount with which the exploration parameter changes after each episode
    temperature = 0.1
    batch_size = 32
    min_size_buffer = 1000
    max_size_buffer = 10000
    exploration_strategy = 'egreedy'
    #exploration_strategy = 'boltzmann'

    start = time.time()

    classical_model = ClassicalModel(n_holes=n_holes, memory_size=memory_size, n_nodes=n_nodes, learning_rate=learning_rate)
    base_model = classical_model.initialize_model()
    target_network = classical_model.initialize_model()

    dqn = DQN(savename, base_model, target_network, n_holes, memory_size, learning_rate, gamma, num_episodes, tau, initial_exploration, final_exploration, decay_constant, temperature, batch_size, min_size_buffer, max_size_buffer, exploration_strategy)

    dqn.main()

    end = time.time()

    print('Total time: {} seconds (number of episodes: {})'.format(round(end - start, 1), num_episodes))

if __name__ == '__main__':
    main()