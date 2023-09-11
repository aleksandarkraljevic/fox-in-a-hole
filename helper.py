import numpy as np

def exponential_anneal(t, start, final, decay_constant):
    ''' 
    Exponential annealing scheduler for epsilon-greedy policy.
    param t:        current timestep
    param start:    initial value
    param final:    value after percentage*T steps
    '''
    return final + (start - final) * np.exp(-decay_constant*t)


def boltzmann_exploration(actions, temperature):
    '''
    Boltzmann exploration policy.
    param actions:      vector with possible actions
    param temperature:  exploration parameter
    return:             vector with probabilities for choosing each option
    '''
    # print(f'bolzmann exploration of {actions}')  # can remove this line once everything works
    actions = actions[0] / temperature  # scale by temperature
    a = actions - max(actions)  # substract maximum to prevent overflow of softmax
    return np.exp(a)/np.sum(np.exp(a))