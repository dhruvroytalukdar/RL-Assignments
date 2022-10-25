import numpy as np
import gym
import random

env = gym.make('Taxi-v3', render_mode='rgb_array')
num_step = 10

# learning rate similar to supervised learning
alpha = 0.15
# parameter on how much to emphasize future gains
gamma = 0.6
# parameter to control exploitation/exploration
epsilon = 0.1

q_table = np.zeros((env.observation_space.n, env.action_space.n))

for i in range(num_step):
    state, info = env.reset()
    print(f"Step {i+1} of {num_step}")

    done = False
    while not done:
        # Choose best possible action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # Do the action
        next_state, reward, done, _, info = env.step(action)

        # Update q-table
        new_q_value = (1-alpha)*q_table[state, action] + \
            alpha*(reward + gamma*np.max(q_table[next_state, :]))

        # Assign new values and go the next state
        q_table[state, action] = new_q_value
        state = next_state

        # env.render()

env.close()
