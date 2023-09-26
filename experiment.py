from dqn import *
from fox_in_a_hole import *

# name that will be used to save both the model and all its data with
savename = 'experiment'
# amount of repetitions that will be averaged over for the experiment
repetitions = 10
# game parameters
n_holes = 5
memory_size = 2*n_holes
# Hyperparameters of the algorithm and other parameters of the program
learning_rate = 0.01
gamma = 1  # discount factor
initial_epsilon = 1  # 100%
final_epsilon = 0.01  # 1%
num_episodes = 1000
decay_constant = 0.1  # the amount with which the exploration parameter changes after each episode
temperature = 0.1
update_freq_TN = memory_size
activate_ER = True
activate_TN = True
exploration_strategy = 'anneal_epsilon_greedy'

data_names = []

start = time.time()

for rep in range(repetitions):
    base_model = initialize_model(n_holes=n_holes, memory_size=memory_size, learning_rate=learning_rate)
    target_network = initialize_model(n_holes=n_holes, memory_size=memory_size, learning_rate=learning_rate)

    episode_lengths, rewards = main(memory_size=memory_size, base_model=base_model, target_network=target_network, num_episodes=num_episodes, gamma=gamma, initial_exploration=initial_epsilon, final_exploration=final_epsilon, learning_rate=learning_rate, decay_constant=decay_constant, temperature=temperature, activate_TN=activate_TN, update_freq_TN=update_freq_TN, activate_ER=activate_ER, exploration_strategy='anneal_epsilon_greedy', n_holes=n_holes)

    data = {'n_holes': n_holes, 'memory_size': memory_size, 'rewards': rewards, 'episode_lengths': episode_lengths, 'activate_ER': activate_ER, 'activate_TN': activate_TN}
    file_name = savename+str(rep)
    np.save('data/'+file_name+'.npy', data)
    target_network.save('models/'+file_name+'.keras')
    data_names.append(file_name)

end = time.time()
print('Total time: {} seconds'.format(round(end - start, 1)))

plot_averaged(data_names=data_names, show=True, savename=savename, smooth=False)