from helper import *

data_names = ['experiment0', 'experiment1', 'experiment2', 'experiment3', 'experiment4', 'experiment5', 'experiment6', 'experiment7', 'experiment8', 'experiment9']
#data_names = ['test']

for name in data_names:
    evaluate(model_name=name, n_samples=10000, print_strategy=True)

#plot(data_name='test', show=True, savename='test', smooth=False)
#plot(data_name='test', show=True, savename='test_smooth', smooth=True)

plot_averaged(data_names=data_names, show=True, savename='smoothed_experiment', smooth=True)