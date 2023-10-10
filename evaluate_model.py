from helper import *

repetitions = 20

data_names = []

for rep in range(repetitions):
    data_names.append('experiment'+str(rep))

#data_names = ['experiment0']

#for name in data_names:
#    evaluate(model_name=name, n_samples=10000, print_strategy=True)

#plot(data_name='test', show=True, savename='test', smooth=False)
#plot(data_name='test', show=True, savename='test_smooth', smooth=True)

plot_averaged(data_names=data_names, show=True, savename='experiment', smooth=False)