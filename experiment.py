from dqn import *

# amount of repetitions that will be averaged over for the experiment
repetitions = 20
# game parameters
n_holes = 5
memory_size = 2*(n_holes-2)
# neural network parameters
n_nodes = 12
# Hyperparameters of the algorithm and other parameters of the program
learning_rate = [0.01]
gamma = 1  # discount factor
initial_exploration = 1  # 100%
final_exploration = 0.01  # 1%
num_episodes = 5000
steps_per_train = 10 # Per how many game steps a singular batch of model training happens
soft_weight_update = True # False if hard updating
steps_per_target_update = 5 # This parameter only matters when the weights are being hard updated
tau = 0.05
decay_constant = [0.01]  # the amount with which the exploration parameter changes after each episode
temperature = 0.1
batch_size = 64
min_size_buffer = 1000
max_size_buffer = 10000
exploration_strategy = 'egreedy'

data_names = []

start = time.time()

for lr in learning_rate:
    for dc in decay_constant:
        savename = 'lr_'+str(lr)+'-dc_'+str(dc)
        for rep in range(repetitions):
            classical_model = ClassicalModel(n_holes=n_holes, memory_size=memory_size, n_nodes=n_nodes,
                                             learning_rate=lr)
            base_model = classical_model.initialize_model()
            target_network = classical_model.initialize_model()

            file_name = savename+'-repetition_'+str(rep+1)

            dqn = DQN(file_name, base_model, target_network, n_holes, memory_size, lr, gamma, num_episodes, steps_per_train, soft_weight_update, steps_per_target_update, tau, initial_exploration, final_exploration, dc, temperature, batch_size, min_size_buffer, max_size_buffer, exploration_strategy)

            dqn.main()

            data_names.append(file_name)

        plot_averaged(data_names=data_names, show=True, savename=savename, smooth=False)
        plot_averaged(data_names=data_names, show=True, savename=savename+'-smooth', smooth=True)

        data_names = []

end = time.time()
print('Total time: {} seconds'.format(round(end - start, 1)))