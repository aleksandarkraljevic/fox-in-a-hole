from helper import *

def evaluate_experiment(parameter_name, repetitions, n_samples, print_strategies):
    data_names = []
    mean_lengths = []

    for rep in range(repetitions):
        data_names.append(parameter_name+'-repetition_'+str(rep+1))

    for name in data_names:
        mean_length = evaluate(model_name=name, n_samples=n_samples, print_strategy=print_strategies)
        mean_lengths.append(mean_length)

    print('Average amount of guesses needed over all repetitions is: ', np.mean(mean_lengths))

def plot_experiment(parameter_name, repetitions, show, save):
    data_names = []

    for rep in range(repetitions):
        data_names.append(parameter_name + '-repetition_' + str(rep + 1))

    if save:
        plot_averaged(data_names=data_names, show=show, savename=parameter_name, smooth=False)
        plot_averaged(data_names=data_names, show=show, savename=parameter_name+'-smooth', smooth=True)
    else:
        plot_averaged(data_names=data_names, show=show, savename=False, smooth=False)
        plot_averaged(data_names=data_names, show=show, savename=False, smooth=True)

def evaluate_model(name, n_samples, print_strategy, plot, show, save):
    evaluate(model_name=name, n_samples=n_samples, print_strategy=print_strategy)
    if plot:
        if save:
            plot(data_name=name, show=show, savename=name, smooth=False)
            plot(data_name=name, show=show, savename=name+'-smooth', smooth=True)
        else:
            plot(data_name=name, show=show, savename=False, smooth=False)
            plot(data_name=name, show=show, savename=False, smooth=True)

parameter_names = ['lr_0.1-temp_0.1', 'lr_0.01-temp_0.1', 'lr_0.001-temp_0.1']
#parameter_name = 'lr_0.0001-temp_0.01'

compare_models(parameter_names=parameter_names, repetitions=20, show=True, savename='compare_best', smooth=True)

#evaluate_experiment(parameter_name=parameter_name, repetitions=20, n_samples=10000, print_strategies=False)

#plot_experiment(parameter_name, 20, True, True)