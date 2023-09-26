import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import savgol_filter
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

def plot(data_name, show, savename, smooth):
    data = np.load('data/'+data_name+'.npy', allow_pickle=True)
    rewards = data.item().get('rewards')
    if smooth==True:
        rewards = savgol_filter(rewards, 5, 1)
    episodes = np.arange(1, len(rewards) + 1)
    plt.figure()
    plt.plot(episodes, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per episode')
    if savename != False:
        plt.savefig('plots/'+savename)
    if show:
        plt.show()

def plot_averaged(data_names, show, savename, smooth):
    n_names = len(data_names)
    data = np.load('data/'+data_names[0]+'.npy', allow_pickle=True)
    rewards = np.asarray(data.item().get('rewards'))
    for i in range(n_names-1):
        new_data =  np.load('data/'+data_names[i+1]+'.npy', allow_pickle=True)
        new_rewards = np.asarray(new_data.item().get('rewards'))
        rewards = np.vstack((rewards, new_rewards))
    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)
    if smooth==True:
        mean_rewards = savgol_filter(mean_rewards, 51, 3)
    episodes = np.arange(1, len(mean_rewards) + 1)
    plt.figure()
    plt.plot(episodes, mean_rewards)
    plt.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, color='gray', alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Mean reward per episode')
    if savename != False:
        plt.savefig('plots/'+savename)
    if show:
        plt.show()
def evaluate(model_name, n_samples, print_strategy):
    model = tf.keras.models.load_model('models/'+model_name+'.keras')
    data = np.load('data/'+model_name+'.npy', allow_pickle=True)
    n_holes = data.item().get('n_holes')
    memory_size = data.item().get('memory_size')
    env = FoxInAHole(n_holes, memory_size)
    observation = [0] * memory_size
    fox, done = env.reset()
    won, lost = done
    current_episode_length = 0
    episode_lengths = []
    rewards = []
    if print_strategy:
        for step in range(len(observation)):
            predicted_q_values = model.predict(np.asarray(observation).reshape(1, memory_size), verbose=0)
            action = np.argmax(predicted_q_values) + 1
            observation[step] = action
        print(observation)
        observation = [0] * memory_size
    for sample in range(n_samples):
        while (not won) and (not lost):
            current_episode_length += 1
            predicted_q_values = model.predict(np.asarray(observation).reshape(1, memory_size), verbose=0)
            action = np.argmax(predicted_q_values) + 1
            reward, done = env.guess(action, current_episode_length)
            won, lost = done
            new_observation = observation.copy()
            new_observation[current_episode_length - 1] = action
            observation = new_observation
            fox = env.step()
        episode_lengths.append(current_episode_length)
        rewards.append(reward)
        current_episode_length = 0
        fox, done = env.reset()
        won, lost = done
        observation = [0] * memory_size
    print('The average amount of guesses needed to finish the game is: ',np.mean(episode_lengths))
    print('The average reward per game after '+str(n_samples)+' games is: ',np.mean(rewards))