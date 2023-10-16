from helper import *

parameter_name = 'lr_0.01-dc_0.01'

repetitions = 20

data_names = []
mean_lengths = []

for rep in range(repetitions):
    data_names.append(parameter_name+'-repetition_'+str(rep+1))

for name in data_names:
    mean_length = evaluate(model_name=name, n_samples=10000, print_strategy=False)
    mean_lengths.append(mean_length)

print('Average amount of guesses needed over all repetitions is: ', np.mean(mean_lengths))

#plot(data_name='test', show=True, savename='test', smooth=False)
#plot(data_name='test', show=True, savename='test_smooth', smooth=True)

#plot_averaged(data_names=data_names, show=True, savename=parameter_name, smooth=False)
#plot_averaged(data_names=data_names, show=True, savename=parameter_name+'-smooth', smooth=True)

#parameter_names = ['lr_0.1-dc_0.01', 'lr_0.01-dc_0.01', 'lr_0.001-dc_0.01']

#compare_models(parameter_names=parameter_names, repetitions=20, show=True, savename='compare_best', smooth=True)