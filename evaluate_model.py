from helper import *

parameter_name = 'lr_0.1-dc_0.1'

repetitions = 20

data_names = []

for rep in range(repetitions):
    data_names.append(parameter_name+'-repetition_'+str(rep+1))

#for name in data_names:
#    evaluate(model_name=name, n_samples=10000, print_strategy=True)

#plot(data_name='test', show=True, savename='test', smooth=False)
#plot(data_name='test', show=True, savename='test_smooth', smooth=True)

plot_averaged(data_names=data_names, show=True, savename=parameter_name, smooth=False)
plot_averaged(data_names=data_names, show=True, savename=parameter_name+'-smooth', smooth=True)

#parameter_names = ['lr_0.0001-dc_0.1', 'lr_0.0001-dc_0.01', 'lr_0.0001-dc_0.001', 'lr_0.0001-dc_0.0001']

#compare_models(parameter_names=parameter_names, repetitions=20, show=True, savename='lr_0.0001', smooth=True)