import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fox_in_a_hole import *
from helper import *

def evaluate(model_name, n_samples, n_holes, print_strategy):
    model = tf.keras.models.load_model(model_name)
    env = FoxInAHole(n_holes, 2 * n_holes)
    observation = [0] * (2 * n_holes)
    fox, done = env.reset()
    won, lost = done
    current_episode_length = 0
    episode_lengths = []
    if print_strategy:
        for step in range(len(observation)):
            predicted_q_values = model.predict(np.asarray(observation).reshape(1, 2 * n_holes), verbose=0)
            action = np.argmax(predicted_q_values) + 1
            observation[step] = action
        print(observation)
        observation = [0] * (2 * n_holes)
    for sample in range(n_samples):
        while (not won) and (not lost):
            current_episode_length += 1
            predicted_q_values = model.predict(np.asarray(observation).reshape(1, 2 * n_holes), verbose=0)
            action = np.argmax(predicted_q_values) + 1
            reward, done = env.guess(action, current_episode_length)
            won, lost = done
            new_observation = observation.copy()
            new_observation[current_episode_length - 1] = action
            observation = new_observation
            fox = env.step()
            if won or lost:
                episode_lengths.append(current_episode_length)
                current_episode_length = 0
                fox, done = env.reset()
                won, lost = done
                observation = [0] * (2 * n_holes)
                break
    print('The average amount of guesses needed to finish the game is: ',np.mean(episode_lengths))

evaluate(model_name='test.keras',n_samples=1000,n_holes=5,print_strategy=True)