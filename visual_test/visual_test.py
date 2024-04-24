import numpy as np
import tensorflow as tf
import math
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings
import pyarrow.parquet as pq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Define the Actor model
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


# Define the Critic model
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

# Define the replay buffer
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

# Define the Agent
class Agent:
    def __init__(self, state_dim, action_dim, action_range, max_buffer_size=10000, batch_size=64, gamma=0.99, tau=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

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
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]

        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

    def choose_action(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = self.actor(state)
        return np.squeeze(action)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

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

        self.update_target_networks()

    def save_weights(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

    def load_actor_weights(self, actor_path):
        self.actor.load_weights(actor_path)

data = """0.390179 1.96935 -0.021415 false
0.375075 1.919374 0.017543 false
0.359967 1.869353 0.053788 false
0.344856 1.819287 0.087318 false
0.32974 1.769175 0.11813 false
0.314619 1.719017 0.146219 false
0.299495 1.668814 0.171585 false
0.284367 1.618564 0.194224 false
0.269234 1.568269 0.214134 false
0.254097 1.517928 0.231314 false
0.238956 1.467541 0.245761 false
0.223811 1.417108 0.257475 false
0.208662 1.366628 0.266455 false
0.193508 1.316103 0.272699 false
0.178351 1.26553 0.276208 false
0.163189 1.214912 0.276981 false
0.148023 1.164246 0.275018 false
0.132853 1.113534 0.270318 false
0.117678 1.062775 0.262882 false
0.1025 1.01197 0.252708 false
0.087317 0.961117 0.239795 false
0.07213 0.910218 0.224141 false
0.056939 0.859271 0.205746 false
0.041744 0.808278 0.184606 false
0.026544 0.757236 0.160721 false
0.011341 0.706148 0.134086 false
-0.003867 0.655012 0.1047 false
-0.019079 0.603829 0.072559 false
-0.034295 0.552598 0.03766 false
-0.049516 0.501319 0.0 true"""

# Split the data into rows
rows = data.split('\n')

# Initialize lists for storing x_pos, y_pos, and z_pos
x_pos = []
y_pos = []
z_pos = []

for row in rows:
    values = row.split()
    x_pos.append(float(values[0]))
    y_pos.append(float(values[1]))
    z_pos.append(float(values[2]))

x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
z_pos = np.array(z_pos)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create an empty scatter plot object for the animation
ball_position, = ax.plot([], [], [], 'bo', markersize=5)
arm_displacement, = ax.plot([], [], [], 'r-')

ax.scatter(x_pos, y_pos, z_pos, color='blue', label='Ball Position')

actor = Actor(9, 4, np.array([0.47124, 0.62832, 0.31416, 0.31416]))

# Call the model on some dummy input to initialize its variables
dummy_input = tf.convert_to_tensor(np.zeros((1, 9)), dtype=tf.float32)
_ = actor(dummy_input)

# Load the actor and use it to predict on unseen data
actor.load_weights("actor_prediction_error.h5")

theta1 =  0.0000000
theta2 = 1.134464
theta3 = 1.7017
theta4 = 1.439901

ball_path = np.column_stack((x_pos,y_pos,z_pos))


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


# Use the actor to predict actions for the new states
arm_x = []
arm_y = []
arm_z = []

prediction_error = ball_path[0] - displacement_function(theta1, theta2, theta3, theta4)

for ball_index in range(0, len(x_pos)):
    curr_state = np.concatenate([ball_path[ball_index].flatten(), displacement_function(theta1, theta2, theta3, theta4), prediction_error])
    action = np.array(actor(np.expand_dims(curr_state, axis=0)))

    theta1 += action[0,0]
    theta2 += action[0,1] 
    theta3 += action[0,2]
    theta4 += action[0,3]

    x,y,z = displacement_function(theta1, theta2, theta3, theta4)
    arm_x = np.append(arm_x, x)
    arm_y = np.append(arm_y, y)
    arm_z = np.append(arm_z, z)

arm_x = np.array(arm_x)
arm_y = np.array(arm_y)
arm_z = np.array(arm_z)

def update(frame):
    if frame < len(arm_x):
        # ball_position.set_data(x_pos[:frame+1], y_pos[:frame+1])
        ball_position.set_data(x_pos[:frame+1], y_pos[:frame+1])
        ball_position.set_3d_properties(z_pos[:frame+1])

        # Update the arm_displacement from origin to the fixed point
        arm_displacement.set_data([0.0, arm_x[frame]], [0.0, arm_y[frame]])
        arm_displacement.set_3d_properties([0.0, arm_z[frame]])

        # Adjust the limits of the axes based on the ball and arm positions
        ax.set_xlim(min(np.min(x_pos), np.min(arm_x)), max(np.max(x_pos), np.max(arm_x)))
        ax.set_ylim(min(np.min(y_pos), np.min(arm_y)), max(np.max(y_pos), np.max(arm_y)))
        ax.set_zlim(min(np.min(z_pos), np.min(arm_z)), max(np.max(z_pos), np.max(arm_z)))

    return ball_position, arm_displacement

# Create the animation
ani = FuncAnimation(fig, update, frames=len(x_pos), blit=True, interval=100)

# Show the plot
plt.show()