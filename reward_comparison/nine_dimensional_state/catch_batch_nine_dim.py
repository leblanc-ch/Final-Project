import numpy as np
import tensorflow as tf
import math
from collections import deque
import random
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import pandas as pd
import os
import math


class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()
        self.action_range = action_range
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.output_layer_1 = tf.keras.layers.Dense(1, activation='tanh')
        self.output_layer_2 = tf.keras.layers.Dense(1, activation='tanh')
        self.output_layer_3 = tf.keras.layers.Dense(1, activation='tanh')
        self.output_layer_4 = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        action_1 = self.output_layer_1(x) * self.action_range[0]
        action_2 = self.output_layer_2(x) * self.action_range[1]
        action_3 = self.output_layer_3(x) * self.action_range[2]
        action_4 = self.output_layer_4(x) * self.action_range[3]
        return tf.concat([action_1, action_2, action_3, action_4], axis=-1)

    def save_actor_weights(self, actor_path):
        self.save_weights(actor_path)

class Critic(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

    def save_critic_weights(self, critic_path):
        self.save_weights(critic_path)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

class Agent:
    def __init__(self, state_dim, action_dim, action_range, max_buffer_size=10000, batch_size=64, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.batch_size = batch_size
        self.gamma = gamma

        # Update the target networks after 200 episodes
        self.update_interval = 200
        self.episode_count = 0

        self.actor = Actor(state_dim, action_dim, action_range)
        self.target_actor = Actor(state_dim, action_dim, action_range)
        self.critic = Critic(state_dim)
        self.target_critic = Critic(state_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.replay_buffer = ReplayBuffer(max_buffer_size)

    def update_target_networks(self):
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        for i in range(len(actor_weights)):
            target_actor_weights[i] = actor_weights[i]

        for i in range(len(critic_weights)):
            target_critic_weights[i] = critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

    def choose_action(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = self.actor(state)
        return np.squeeze(action)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    @tf.function
    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_next_Q = tf.squeeze(self.target_critic(next_states, target_actions), 1)
            target_Q = rewards + self.gamma * (1 - dones) * target_next_Q
            predicted_Q = tf.squeeze(self.critic(states, actions), 1)
            critic_loss = tf.keras.losses.MSE(target_Q, predicted_Q)

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, new_actions))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Episode counter
        self.episode_count += 1

        # Check to see if the target networks need updates
        if self.episode_count % self.update_interval == 0:
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())

    def save_weights(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

    # The reward function for training is more strict, this is more lax
# This function represents the tennis ball as a 3d object, and not a 1d point
# Because it has a surface, it's much easier to collide with
def reward_function_test(ball_position, arm_position):
    euc_dist = np.linalg.norm(ball_position - arm_position)

    tennis_ball_radius_meters = 0.032

    if euc_dist <= tennis_ball_radius_meters:
       return True
    else:
       return False

def displacement_function(theta_1, theta_2, theta_3, theta_4):
    # the links represent the length of the joints in meters
    link_1 = .115
    link_2 = .2285
    link_3 = .2285
    link_4 = .24

    rotation_1_to_2 = np.array([[math.cos(theta_1), -math.sin(theta_1), 0],
                            [math.sin(theta_1),  math.cos(theta_1), 0],
                            [0,0,1]
                            ])

    rotation_2_to_3 = np.array([[1,0,0],
                            [0, math.cos(theta_2), -math.sin(theta_2)],
                            [0, math.sin(theta_2), math.cos(theta_2)]
                            ])

    rotation_3_to_4 = np.array([[1,0,0],
                            [0, math.cos(theta_3), -math.sin(theta_3)],
                            [0, math.sin(theta_3), math.cos(theta_3)]
                            ])

    rotation_4_to_5 = np.array([[1,0,0],
                            [0, math.cos(theta_4), -math.sin(theta_4)],
                            [0, math.sin(theta_4), math.cos(theta_4)]
                            ])

    displace_1_to_2 = np.array([[0],[0],[link_1]])

    H_1_2 = np.concatenate((rotation_1_to_2, displace_1_to_2), 1)
    H_1_2 = np.concatenate((H_1_2, [[0,0,0,1]]), 0)

    displace_2_to_3 = np.array([[0],
                    [link_2 * math.sin(theta_2)],
                    [link_2 * math.cos(theta_2)]
                    ])

    H_2_3 = np.concatenate((rotation_2_to_3, displace_2_to_3), 1)
    H_2_3 = np.concatenate((H_2_3, [[0,0,0,1]]), 0)

    displace_3_to_4 = np.array([[0],
                    [link_3 * math.sin(theta_3)],
                    [link_3 * math.cos(theta_3)]
                    ])

    H_3_4 = np.concatenate((rotation_3_to_4, displace_3_to_4), 1)
    H_3_4 = np.concatenate((H_3_4, [[0,0,0,1]]), 0)

    displace_4_to_5 = np.array([[0],
                    [link_4 * math.sin(theta_4)],
                    [link_4 * math.cos(theta_4)]
                    ])

    H_4_5 = np.concatenate((rotation_4_to_5, displace_4_to_5), 1)
    H_4_5 = np.concatenate((H_4_5, [[0,0,0,1]]), 0)

    return np.dot(np.dot(np.dot(H_1_2, H_2_3), H_3_4), H_4_5)[0:3,3]

# Read the Parquet file
table = pq.read_table('model_test_data_1000.parquet')

'''
    Column names of the data are : Xpos
                                   Ypos
                                   Zpos
                                   DoneFlag
'''

# Convert the table to a pandas DataFrame
test_dataset = table.to_pandas()

# Get however many episodes there are in the data
number_episodes = test_dataset[test_dataset['DoneFlag'] == True].shape[0]

# Remove any row that has an illegal Zpos
filtered_data = test_dataset[test_dataset['Zpos'] >= 0.000000].copy()

folder_path = os.getcwd()

actor_files = [file for file in os.listdir(folder_path) if file.endswith('.h5')]

# Load the weights that correspond to the actor that's going to be tested
# Load the weights that correspond to the actor that's going to be tested
actor_one = Actor(9, 4, np.array([0.47124, 0.62832, 0.31416, 0.31416]))
dummy_input = tf.convert_to_tensor(np.zeros((1, 9)), dtype=tf.float32)
_ = actor_one(dummy_input)


best_caught = 0
best_index = 0

for file_index, file_name in enumerate(actor_files):
    file_path = os.path.join(folder_path, file_name)
    actor_one = Actor(9, 4, np.array([0.47124, 0.62832, 0.31416, 0.31416]))

    dummy_input = tf.convert_to_tensor(np.zeros((1, 9)), dtype=tf.float32)
    _ = actor_one(dummy_input)
    actor_one.load_weights(file_path)

    theta1 =  0.0000000
    theta2 = 1.134464
    theta3 = 1.7017
    theta4 = 1.439901

    number_caught = 0

    file_error = 0

    ball_index = 0

    while ball_index + 1 < filtered_data.shape[0]:

        # current_ball_position = filtered_data.iloc[ball_index:ball_index+3, 0:3].to_numpy().flatten()

        # curr_state_one = np.concatenate([current_ball_position, displacement_function(theta1, theta2, theta3, theta4)])
        ball_position = filtered_data.iloc[ball_index, 0:3].to_numpy().flatten()
        arm_position = displacement_function(theta1, theta2, theta3, theta4)
        prediction_error = ball_position - arm_position

        curr_state = np.concatenate((ball_position, arm_position, prediction_error)).astype(np.float32)

        # Reshape curr_state to match the expected input shape of the model
        curr_state_reshaped = np.expand_dims(curr_state, axis=0)

        # Pass the reshaped input to the model for inference
        action = actor_one(curr_state_reshaped)

        # action = agent.choose_action(state)

        theta1 += action[0,0]
        theta2 += action[0,1]
        theta3 += action[0,2]
        theta4 += action[0,3]

        theta1 = np.clip(theta1, -1.0472, 1.0472)
        theta2 = np.clip(theta2, .785398, 1.48353)
        theta3 = np.clip(theta3, 1.5708, 1.8326)
        theta4 = np.clip(theta4, 1.047202, 1.8326)

        ball_was_caught = reward_function_test(curr_state[0:3], curr_state[3:6])

        file_error += np.linalg.norm(curr_state[0:3] - curr_state[3:6])

        if ball_was_caught:
            number_caught+=1

            # print(number_caught)

            #Reset the arm to intial conditions
            theta1 =  0.0000000
            theta2 = 1.134464
            theta3 = 1.7017
            theta4 = 1.439901

            # Selects the data from the current position to the end
            data_here_to_the_end = filtered_data.iloc[ball_index:]

            #This gives the index of the first item in the sliced array that signals the end of the current episode
            true_index_current_episode = data_here_to_the_end[data_here_to_the_end['DoneFlag']].index[0]

            # This moves the index to the start of a new episode
            ball_index = true_index_current_episode + 1




        else:
            ball_index+=1

            new_ball_position = filtered_data.iloc[ball_index, 0:3].to_numpy().flatten()
            new_arm_position = displacement_function(theta1, theta2, theta3, theta4)
            new_prediction_error = new_ball_position - new_arm_position

            next_state =  np.concatenate((new_ball_position, new_arm_position, new_prediction_error)).astype(np.float32)

            state = next_state

            # Checks for the need to skip to the next episode, if so reset to the initial arm state
            if filtered_data.iloc[ball_index, 3] == True:
            #move on to the next episode, skip straight to the first false after this true value
                ball_index += 1

                theta1 =  0.0000000
                theta2 = 1.134464
                theta3 = 1.7017
                theta4 = 1.439901



    if number_caught > best_caught:
        best_caught = number_caught
        best_index = file_index

    print("file : {} with error: {}".format(file_name, file_error))

print("best accuracy : {} % from file  {}".format((best_caught/number_episodes)*100, actor_files[best_index]))