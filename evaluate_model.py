from helper import *

data_names = ['experiment0', 'experiment1', 'experiment2', 'experiment3', 'experiment4']

for name in data_names:
    evaluate(model_name=name, n_samples=1, print_strategy=True)

#plot(data_name='test2', show=True, savename='test2')

#plot_averaged(data_names=data_names, show=True, savename='smoothed_experiment', smooth=True)