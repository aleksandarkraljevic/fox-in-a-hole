from helper import *

#evaluate(model_name='test2', n_samples=500, print_strategy=True)

#plot(data_name='test2', show=True, savename='test2')

data_names = ['experiment0', 'experiment1', 'experiment2', 'experiment3', 'experiment4', 'experiment5', 'experiment6', 'experiment7', 'experiment8', 'experiment9']
plot_averaged(data_names=data_names, show=True, savename='smoothed_experiment', smooth=True)