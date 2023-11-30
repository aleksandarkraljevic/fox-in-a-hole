from tqdm import tqdm
import time
import random

from helper import *
from dnn import *


class DDQN():
    def __init__(self, savename, base_model, target_network, n_holes, memory_size, learning_rate, gamma, num_episodes, steps_per_train, soft_weight_update, steps_per_target_update, tau, initial_exploration, final_exploration, decay_constant, temperature, batch_size, min_size_buffer, max_size_buffer, exploration_strategy):
        '''
        Initializes the DDQN parameters.

        Parameters
        ----------
        savename (str):
            The name with which the file model and data files will be saved.
        base_model (tensorflow keras model):
            The base network.
        target_model (tensorflow keras model):
            The target network.
        n_holes (int):
            Number of outputs of the DNN. / Number of holes in the environment.
        memory_size (int):
            The size of the input to the DNN. / The amount of guesses that the agent is allowed to look back.
        learning_rate (float):
            The learning rate that the optimizer of the DNN uses in order to update the weights.
        gamma (float):
            The discount factor that is used in the Q-learning algorithm.
        num_episodes (int):
            The amount of total episodes that the model will train for.
        steps_per_train (int):
            The amount of time steps that pass in between each training step.
        soft_weight_update (boolean):
            Whether the target network will be updated via soft-updating. If False, then it will be hard-updating.
        steps_per_target_update (int):
            Per how many training steps the target network will update, if it is hard-updating.
        tau (float):
            The fraction with which the base network copies over to the target network after each training step.
        initial_exploration (float):
            The starting value of epsilon in the case that annealing epsilon-greedy is used.
        final_exploration (float):
            The lowest value of epsilon in the case that annealing epsilon-greedy is used.
        decay_constant (float):
            How fast epsilon decays in the case that annealing epsilon-greedy is used.
        temperature (float):
            The strength with which exploration finds place in the case that the Boltzmann policy is used.
        batch_size (int):
            The amount of samples that each training batch consists of.
        min_size_buffer (int):
            The minimum size that the experience replay buffer needs to be before training may start.
        max_size_buffer (int):
            The maximum size that the experience replay buffer is allowed to be. If this limit is reached then the oldest samples start being replaced with the newest samples.
        exploration_strategy (str):
            What exploration strategy should be followed during the training of the model. Either "egreedy" or "boltzmann".
        '''
        self.savename = savename
        self.n_holes = n_holes
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.base_model = base_model
        self.target_network = target_network
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.steps_per_train = steps_per_train
        self.soft_weight_update = soft_weight_update
        self.steps_per_target_update = steps_per_target_update
        self.tau = tau
        self.initial_exploration = initial_exploration
        self.final_exploration = final_exploration
        self.decay_constant = decay_constant
        self.temperature = temperature
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.max_size_buffer = max_size_buffer
        self.exploration_strategy = exploration_strategy

    def soft_update_model(self):
        '''
        Copies weights from the base network to the target network via a soft-update rule.
        '''
        new_weights = []
        for TN_layer, BM_layer in zip(self.target_network.get_weights(), self.base_model.get_weights()):
            new_weights.append((1 - self.tau) * TN_layer + self.tau * BM_layer)
        self.target_network.set_weights(new_weights)

    def hard_update_model(self):
        '''
        Copies weights from the base network to the target network via a hard-update rule.
        '''
        self.target_network.set_weights(self.base_model.get_weights())

    def save_data(self, rewards, episode_lengths):
        '''
        Saves the model after its training, as well as important results and properties.

        Parameters
        ----------
        rewards (list):
            A list of all the rewards that were obtained at the end of each episode.
        episode_lengths (list):
            A list of the length of each episode.
        '''
        data = {'n_holes': self.n_holes, 'rewards': rewards, 'episode_lengths': episode_lengths}
        np.save('data/' + self.savename + '.npy', data)
        self.target_network.save('models/' + self.savename + '.keras')

    def custom_predict(self, observations, model, batch_size):
        '''
        A custom keras model predict function that can predict a batch of samples.

        Parameters
        ----------
        observations (array):
            An array containing each sample's action, state, next state, reward, and whether it finished the game or not.
        model (tensorflow keras model):
            The model that is used to make predictions with.
        batch_size (int):
            The amount of samples that each training batch consists of.
        '''
        predicted_q_values = []
        for observation in observations:
            predicted_q_values.append(model(np.asarray(observation).reshape(1,self.memory_size)))
        return np.reshape(predicted_q_values, (batch_size,self.n_holes))

    def train(self, replay_buffer):
        '''
        Trains the DDQN model.

        Parameters
        ----------
        replay_buffer (array):
            An array containing all the samples that can be used to train with.
        '''

        sample_list = random.sample(range(0, len(replay_buffer)), self.batch_size)

        observation_list = list()
        new_observation_list = list()
        action_list = list()
        reward_list = list()
        done_list = list()
        for element in sample_list:
            observation_list.append(replay_buffer[element][0])
            new_observation_list.append(replay_buffer[element][3])
            action_list.append(replay_buffer[element][1])
            reward_list.append(replay_buffer[element][2])
            done_list.append(replay_buffer[element][4])

        predicted_q_values = self.custom_predict(observation_list, self.base_model, self.batch_size)

        new_predicted_q_values = self.custom_predict(new_observation_list, self.target_network, self.batch_size)

        for i in range(len(observation_list)):
            target_q = reward_list[i] + self.gamma * max(new_predicted_q_values[i]) * (1 - float(done_list[i]))
            # here we have to replace the q value of the hole that we ended up choosing with the target q value so that only that one gets updated
            predicted_q_values[i,action_list[i]-1] = target_q
            target_q_values = predicted_q_values # rename it just for clarity's sake

        self.base_model.fit(x=np.asarray(observation_list), y=target_q_values, batch_size=self.batch_size, verbose=0)

    def main(self):
        '''
        Handles the main bulk of the DDQN algorithm, making use of all the other functions in this class.
        '''

        env = FoxInAHole(self.n_holes)

        episode_lengths = []
        episode_rewards = []
        replay_buffer = deque(maxlen=self.max_size_buffer)
        total_steps = 0
        training_steps = 0
        possible_actions = list(range(self.n_holes))
        possible_actions = [x+1 for x in possible_actions]

        self.hard_update_model() # start by copying over the weights from TN to base model to ensure they are identical

        for episode in tqdm(range(self.num_episodes)):
            done = env.reset()
            observation = deque([0]*self.memory_size, maxlen=self.memory_size)  # The memory of actions that have been taken is the observation
            episode_reward = 0
            current_episode_length = 0

            if self.exploration_strategy == 'egreedy':
                # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
                exploration_parameter = exponential_anneal(episode, self.initial_exploration, self.final_exploration, self.decay_constant)

            while not done:
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

                reward, done = env.guess(action)
                new_observation = observation.copy()
                new_observation.append(action)
                replay_buffer.append([observation, action, reward, new_observation, done])
                episode_reward += reward

                if (total_steps % self.steps_per_train == 0) and (len(replay_buffer) > self.min_size_buffer): # the model is trained after every game, as long as the replay buffer is filled up enough
                    self.train(replay_buffer)
                    training_steps += 1

                if self.soft_weight_update:
                    self.soft_update_model()  # copy over part of the weights at each step (soft update)
                elif training_steps % self.steps_per_target_update == 0: # copy over the weights only after a certain amount of training steps have been taken (hard update)
                    self.hard_update_model()

                # roll over
                observation = new_observation
                env.step()

            episode_lengths.append(current_episode_length)
            episode_rewards.append(episode_reward)

        if self.savename != False:
            self.save_data(episode_rewards, episode_lengths)

def main():
    '''
    Initializes all the hyperparameters, creates the base and target network by calling upon dnn.py, and trains and saves the model by calling upon the DQN() class.
    '''
    # name that will be used to save both the model and all its data with
    savename = 'test'
    # game parameters
    n_holes = 5
    memory_size = 2*(n_holes-2)
    # model hyperparameters
    hidden_layers = 2
    n_nodes = 12
    # hyperparameters of the algorithm and other parameters of the program
    learning_rate = 0.01
    gamma = 1  # discount factor
    steps_per_train = 5 # Per how many game steps a singular batch of model training happens
    soft_weight_update = True # False if hard updating
    steps_per_target_update = 5 # This parameter only matters when the weights are being hard updated
    tau = 0.05 # TN soft-update speed parameter, tau is the ratio of the TN that gets copied over at each training step
    initial_exploration = 1  # 100%
    final_exploration = 0.01  # 1%
    num_episodes = 5000
    decay_constant = 0.01  # the amount with which the exploration parameter changes after each episode
    temperature = 0.1
    batch_size = 64
    min_size_buffer = 1000
    max_size_buffer = 10000
    #exploration_strategy = 'egreedy'
    exploration_strategy = 'boltzmann'

    start = time.time()

    classical_model = ClassicalModel(n_holes=n_holes, memory_size=memory_size, hidden_layers=hidden_layers, n_nodes=n_nodes, learning_rate=learning_rate)
    base_model = classical_model.initialize_model()
    target_network = classical_model.initialize_model()

    ddqn = DDQN(savename, base_model, target_network, n_holes, memory_size, learning_rate, gamma, num_episodes, steps_per_train, soft_weight_update, steps_per_target_update, tau, initial_exploration, final_exploration, decay_constant, temperature, batch_size, min_size_buffer, max_size_buffer, exploration_strategy)

    ddqn.main()

    end = time.time()

    print('Total time: {} seconds (number of episodes: {})'.format(round(end - start, 1), num_episodes))

if __name__ == '__main__':
    main()