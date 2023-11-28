from helper import *

def evaluate_experiment(parameter_name, repetitions, n_samples, print_strategies, print_evaluation):
    mean_lengths = []

    for rep in range(repetitions):
        name = parameter_name+'-repetition_'+str(rep+1)
        print('For repetition '+str(rep+1)+':')
        mean_length = evaluate(model_name=name, n_samples=n_samples, print_strategy=print_strategies, print_evaluation=print_evaluation, plot_distribution=False, save=False)
        mean_lengths.append(mean_length)

    print('Average amount of guesses needed over all repetitions is:', round(np.mean(mean_lengths),2), '+- ', round(np.std(mean_lengths)/np.sqrt(repetitions),2))

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

def evaluate_model(name, n_samples, print_strategy, print_evaluation, plot_model, show, save, plot_distribution):
    evaluate(model_name=name, n_samples=n_samples, print_strategy=print_strategy, print_evaluation=print_evaluation, plot_distribution=plot_distribution, save=save)
    if plot_model:
        if save:
            plot(data_name=name, show=show, savename=name, smooth=False)
            plot(data_name=name, show=show, savename=name+'-smooth', smooth=True)
        else:
            plot(data_name=name, show=show, savename=False, smooth=False)
            plot(data_name=name, show=show, savename=False, smooth=True)

parameter_names = ['temp_0.01', 'lr_out_0.1']
label_names = [r'$\tau$ = 0.01', 'd = 0.01']
parameter_name = 'holes_8'

evaluate_model('holes_8-repetition_6', 10000, False, False, False, False, True, True)

#compare_models(parameter_names=parameter_names, repetitions=20, show=True, savename='compare-exploration', label_names=label_names, smooth=True)

#_experiment(parameter_name=parameter_name, repetitions=20, n_samples=10000, print_strategies=True, print_evaluation=True)

#plot_experiment(parameter_name, 20, True, True)