import numpy as np
import tensorflow as tf
import math
from collections import deque
import random

# Dat Boot
random.seed(225)

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

        # Update the target networks after 2000 episodes
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

            # I changed this to assign value based on the current state rather than the next state, this NEEDS to change back to the way that it was for this training function to work
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

# This is the function used to measure the distance from the base of the robot to the end effector (arm claw thingy)
@tf.function
def displacement_function(theta_1, theta_2, theta_3, theta_4):
    link_1 = 0.115
    link_2 = 0.2285
    link_3 = 0.2285
    link_4 = 0.24

    rotation_1_to_2 = tf.convert_to_tensor([[tf.cos(theta_1), -tf.sin(theta_1), 0],
                                            [tf.sin(theta_1), tf.cos(theta_1), 0],
                                            [0, 0, 1]], dtype=tf.float32)

    rotation_2_to_3 = tf.convert_to_tensor([[1, 0, 0],
                                            [0, tf.cos(theta_2), -tf.sin(theta_2)],
                                            [0, tf.sin(theta_2), tf.cos(theta_2)]], dtype=tf.float32)

    rotation_3_to_4 = tf.convert_to_tensor([[1, 0, 0],
                                            [0, tf.cos(theta_3), -tf.sin(theta_3)],
                                            [0, tf.sin(theta_3), tf.cos(theta_3)]], dtype=tf.float32)

    rotation_4_to_5 = tf.convert_to_tensor([[1, 0, 0],
                                            [0, tf.cos(theta_4), -tf.sin(theta_4)],
                                            [0, tf.sin(theta_4), tf.cos(theta_4)]], dtype=tf.float32)

    displace_1_to_2 = tf.convert_to_tensor([[0.0], [0.0], [link_1]], dtype=tf.float32)

    H_1_2 = tf.concat([rotation_1_to_2, displace_1_to_2], axis=1)
    H_1_2 = tf.concat([H_1_2, tf.convert_to_tensor([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)], axis=0)

    displace_2_to_3 = tf.convert_to_tensor([[0.0],
                                             [link_2 * tf.sin(theta_2)],
                                             [link_2 * tf.cos(theta_2)]], dtype=tf.float32)

    H_2_3 = tf.concat([rotation_2_to_3, displace_2_to_3], axis=1)
    H_2_3 = tf.concat([H_2_3, tf.convert_to_tensor([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)], axis=0)

    displace_3_to_4 = tf.convert_to_tensor([[0.0],
                                             [link_3 * tf.sin(theta_3)],
                                             [link_3 * tf.cos(theta_3)]], dtype=tf.float32)

    H_3_4 = tf.concat([rotation_3_to_4, displace_3_to_4], axis=1)
    H_3_4 = tf.concat([H_3_4, tf.convert_to_tensor([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)], axis=0)

    displace_4_to_5 = tf.convert_to_tensor([[0.0],
                                             [link_4 * tf.sin(theta_4)],
                                             [link_4 * tf.cos(theta_4)]], dtype=tf.float32)

    H_4_5 = tf.concat([rotation_4_to_5, displace_4_to_5], axis=1)
    H_4_5 = tf.concat([H_4_5, tf.convert_to_tensor([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)], axis=0)

    return tf.matmul(tf.matmul(tf.matmul(H_1_2, H_2_3), H_3_4), H_4_5)[:3, 3]


# this function is to both give reward and tell the agent that it's time to stop tracking this episode
def reward_function(ball_position, arm_position):
    euc_dist = np.linalg.norm(ball_position - arm_position)

    if euc_dist < 0.001:
        return 0.0, True
    else:
        return -euc_dist, False

# this is the function used in the gradient ascent portion, used to tell if the reward is increasing with changes in the thetas
def reward_function_grad(ball_position, arm_position):
    euc_dist = tf.norm(ball_position - arm_position)

    if euc_dist < 0.001:
        return tf.constant(0.0)
    else:
        return -euc_dist

def param_initialize(theta_one, theta_two, side_length):
    # Center angle is given in degrees
    center_angle = theta_two-theta_one

    # Value is given by the law of sines
    flat_top_length = (side_length * math.sin(math.radians(center_angle)))/(math.sin(math.radians(theta_one)))

    # All lengths are given in centimeters
    y_min = 0
    y_max = flat_top_length
    y_mid = flat_top_length/2

    # y_max is also given by the law of sines, height of the triangle from the point to flat_top
    x_max = 2 * (y_mid/math.sin(math.radians(center_angle/2)))*(math.sin(math.radians(theta_one)))
    x_min = 0

    right_slope = y_max/((x_max/2))
    left_slope = -right_slope

    left_intercept = y_max - (left_slope * x_min)
    right_intercept = y_max - (right_slope * x_max)

    return x_min, x_max, y_min, y_max, left_slope, left_intercept, right_slope, right_intercept

#if the position is outside of the effective range of the arm, quit recording the trajectory
def within_range(Xpos, Ypos, x_max, y_max):
    #euc_dis_from_origin = math.sqrt((Xpos**2) + (Ypos**2) + (Zpos**2))

    #if euc_dis_from_origin > radius:
    if abs(Xpos) > x_max or abs(Ypos) > y_max:
        return False
    else:
        return True

'''
     My GPU gets mad at me if I make this a tf function, so this will not be a tf function
'''
def theta_gradient(ball_loc, learning_rate):
    with tf.GradientTape() as tape:
        displacement = displacement_function(theta1, theta2, theta3, theta4)
        optimized_reward = reward_function_grad(ball_loc, displacement)

    gradients = tape.gradient(optimized_reward, [theta1, theta2, theta3, theta4])

    # Clipping the gradients to be within about 1.8 degree +or- from their current expression, want to incentivize small changes
    # list on the right is the max_change for each of the thetas, didn't want to use global variables, might modify this later
    clipped_gradients = [tf.clip_by_value(grad, -max_change, max_change) for grad, max_change in zip(gradients, [0.0314159, 0.0314159, 0.0314159, 0.0314159])]

    # Update thetas to enforce maximum amount of movement allowed on one time step
    # Want to minimize negative reward, so maximize, do Gradient Ascent
    theta1.assign_add(learning_rate * clipped_gradients[0])
    theta2.assign_add(learning_rate * clipped_gradients[1])
    theta3.assign_add(learning_rate * clipped_gradients[2])
    theta4.assign_add(learning_rate * clipped_gradients[3])

    # Make sure thetas don't exceed the bounds of where they're allowed to bend

    # max_theta1 = 1.0472
    # min_theta1 = -1.0472
    # max_theta2 = 1.48353
    # min_theta2 = .785398
    # max_theta3 = 1.8326
    # min_theta3 = 1.5708
    # max_theta4 = 1.8326
    # min_theta4 = 1.047202
    # Again, trying not to use global variables, in order to speed up the code

    theta1.assign(tf.clip_by_value(theta1, -1.0472, 1.0472))
    theta2.assign(tf.clip_by_value(theta2, .785398, 1.48353))
    theta3.assign(tf.clip_by_value(theta3, 1.5708, 1.8326))
    theta4.assign(tf.clip_by_value(theta4, 1.047202, 1.8326))

    return optimized_reward

# returns a numpy array that describes a tennis ball trajectory, with drag
@tf.function
def ball_trajectory(x_min, x_max, y_min, y_max, semi_circ_prob, left_slope, right_slope, left_intercept, right_intercept):
    area_prob = random.uniform(0,1)

    x_pos_sample, y_pos_sample, z_pos_sample = [],[],[]

    if area_prob > semi_circ_prob:
        # generate random (x,y) values somewhere within rectangle
        x_rand = random.uniform(x_min, x_max)
        y_rand = random.uniform(y_min, y_max)

        # y output of the line that runs from (x_min,y_min) to (x_max, y_max)
        y_out_left = (left_slope * x_rand) + left_intercept
        y_out_right = (right_slope * x_rand) + right_intercept

        # this is hard to describe, plot out in 2d to explain
        if x_rand < (x_max/2) and y_rand < y_out_left:
            # distance of the randomly generated point to the midpoint of the height
            dist_y_to_mid_height_left = abs(y_rand - (y_max/2))

            #reflect over mid line(height)
            y_rand = y_rand + dist_y_to_mid_height_left

            # x coordinate given by the left line
            x_left = (y_rand - left_intercept) / (left_slope)
            x_left_dist = abs(x_rand - x_left)

            # reflect over mid line(width)
            x_rand = x_rand + x_left_dist

        elif x_rand > (x_max/2) and y_rand < y_out_right:
            # distance of the randomly generated point to the midpoint of the height
            dist_y_to_mid_height_right = abs(y_rand - (y_max/2))

            #reflect over mid line(height)
            y_rand = y_rand + dist_y_to_mid_height_right

            # x coordinate given by the right line
            x_right = (y_rand - right_intercept) / (right_slope)
            x_right_dist = abs(x_rand - x_right)

            # reflect over mid line(width)
            x_rand = x_rand - x_right_dist

        if y_rand > y_max:
            y_minus = 2 * (y_rand - y_max)
            y_rand = y_rand - y_minus
    else:
        rand_angle = random.uniform(0,180)
        rand_radius  = random.uniform(0,(x_max/2))

        x_rand = x_max/2
        y_rand = y_max

        x_add = rand_radius * (math.sin(math.radians(rand_angle)))
        y_add = rand_radius * (math.sin(math.radians(90-rand_angle)))

        if rand_angle < 90:
            x_rand = x_rand + x_add
            y_rand = y_rand + y_add
        else:
            x_rand = x_rand - x_add
            y_rand = y_rand - y_add

    # Initial conditions of the thrown ball
    initial_velocity = random.uniform(2,7)  # (m/s)

    # angles relative to x plane (radians), y plane (radians), and z plane (radians)
    psi = np.float32(np.radians(random.uniform(1,89)))
    theta = np.float32(np.radians(random.uniform(1,89)))
    phi = np.float32(np.radians(random.uniform(1,89)))

    # meters/second for all of these
    initial_x_velocity = np.float32(initial_velocity * np.cos(psi))
    initial_y_velocity = np.float32(initial_velocity * np.cos(theta))
    initial_z_velocity = np.float32(initial_velocity * np.cos(phi))

    # shift the x position over, so that the tip of the triangle is the origin of the frame, then convert the centimeters to meters
    x_position = np.float32((x_rand - (x_max/2))/100)

    # convert the centimeters to meters for the y position
    y_position = np.float32(y_rand/100)

    # the ball starts on the ground, so z = 0
    z_position = np.float32(0)

    x_pos_sample.append(x_position)
    y_pos_sample.append(y_position)
    z_pos_sample.append(z_position)


    # Lists to store the trajectory
    #x_vals, y_vals, z_vals = [x_position], [y_position], [z_position]
    x_velocity, y_velocity, z_velocity = initial_x_velocity, initial_y_velocity, initial_z_velocity

    time_steps_sample = 1

    # Simulation loop for a single sample (one throw of the ball)
    # the average number of time steps tends to be around 28, letting it run for 30 times steps allows it to be a nice number
    # this gives the agent about half a second maximum to catch the ball once it enters within its range
    #while z_position >= 0  and time_steps_sample < 30 and within_range(x_position, y_position, z_position, sphere_radius):
    while z_position >= 0 and within_range(x_position, y_position, x_max, y_max):

        time_steps_sample+=1

        # Calculate change in acceleration due to drag
        #ax_drag, ay_drag, az_drag = drag_acceleration(x_velocity, y_velocity, z_velocity, drag_force)

        # Update drag
        drag_x = drag_force * x_velocity * abs(x_velocity)
        drag_y = drag_force * y_velocity * abs(y_velocity)
        drag_z = drag_force * z_velocity * abs(z_velocity)

        # Update Acceleration
        ax_drag = drag_x / m
        ay_drag = drag_y / m
        az_drag = (drag_z / m) - g

        # Update velocities
        x_velocity += ax_drag * dt
        y_velocity += ay_drag * dt
        z_velocity += az_drag * dt

        # Update position
        x_position += x_velocity * dt
        y_position += y_velocity * dt
        z_position += z_velocity * dt

        # Store position
        x_pos_sample.append(x_position)
        y_pos_sample.append(y_position)
        z_pos_sample.append(z_position)

    x_pos_sample.reverse()
    y_pos_sample.reverse()
    z_pos_sample.reverse()

    return np.column_stack((x_pos_sample, y_pos_sample, z_pos_sample))


x_min, x_max, y_min, y_max, left_slope, left_intercept, right_slope, right_intercept = param_initialize(30, 150, 68.35)

triangle_area = y_max * x_max * 0.5
semi_circ_area = math.pi * ((x_max/2) ** 2) * 0.5
total_area = triangle_area + semi_circ_area
semi_circ_prob = semi_circ_area / total_area

# convert from centimeters to meters
sphere_radius = np.float32((y_max + (x_max/2))/100)

#These values won't change from simulation to simulation

# gravity (m/s^2)
g = 9.81

# air density (at 70F) (kg/m^3)  
rho = 1.20 

# drag coefficient (experimentally found average for tennis balls)
Cd = 0.507  

# cross-sectional area (value for the balls tested) (m^2)
A = 0.0034  

# mass (assumption based on looking up tennis ball stats) (kg)
m = 0.057   

# time-step (based on 60fps frame rate of camera) (s)
dt = 0.0167

# drag force is in opposition to the direction of the ball
drag_force = -0.5 * Cd * A * rho

# ball, arm, error in model prediction : (3,3,3)
state_dim =  9

# The 4 actuators are all move independently from each other, movement from all 4 motors constitutes one action
action_dim = 4

# the action range needs to be the max limit that the actuator can move in the time step, otherwise the arm would just teleport to the ball
# the action range is the choice at this time step, but the sum of the individual actions is how they're expressed in the state
action_range = np.array([0.47124, 0.62832, 0.31416, 0.31416])

agent = Agent(state_dim, action_dim, action_range)

number_of_trials = 100000

epsilon = 1.0  
epsilon_min = 0.01  
epsilon_decay = 0.99995

max_theta1 = 1.0472
min_theta1 = -1.0472
max_theta2 = 1.48353
min_theta2 = .785398
max_theta3 = 1.8326
min_theta3 = 1.5708
max_theta4 = 1.8326
min_theta4 = 1.047202

# Training loop
for episode in range(number_of_trials):

    ball_path = ball_trajectory(x_min, x_max, y_min, y_max, semi_circ_prob, left_slope, right_slope, left_intercept, right_intercept)

    ball_index = 0

    #Starting angles given in radians, placed at the mid point of the limit of their bounds for each of the actuators
    theta1 =  0.0000000
    theta2 = 1.134464
    theta3 = 1.7017
    theta4 = 1.439901

    prediction_error = np.array([0,0,0])

    state = np.concatenate((ball_path[ball_index], displacement_function(theta1, theta2, theta3, theta4), prediction_error)).astype(np.float32)

    done = False
   
    if episode % 1000 == 0:
      # The real dummy is tensorflow, saving weights shouldn't be this hard
      dummy_state = tf.zeros((1, state_dim))
      dummy_action = tf.zeros((1, action_dim))
      _ = agent.critic(dummy_state, dummy_action)
      agent.actor.save_actor_weights("actor_epsilon_error{}.h5".format(episode))
      agent.critic.save_critic_weights("critic_epsilon_error{}.h5".format(episode))

    epsilon *= epsilon_decay
    epsilon = max(epsilon, epsilon_min)

    while not done and ball_index + 1 < ball_path.shape[0]:

        '''
        Do not comment this out, ever, or you'll spend several hours wondering why your model isn't saving its weights
        '''
        action = agent.choose_action(state)

       
        # Epsilon Random Action Selection
        if np.random.rand() < epsilon:
          action1 = np.random.uniform(low=-0.47124, high=0.47124)
          action2 = np.random.uniform(low=-0.62832, high=0.62832)
          action3 = np.random.uniform(low=-0.31416, high=0.31416)
          action4 = np.random.uniform(low=-0.31416, high=0.31416)

          action = np.array([action1, action2, action3, action4]).flatten()

        theta1 += action[0]
        theta2 += action[1]
        theta3 += action[2]
        theta4 += action[3]

        theta1 = np.clip(theta1, -1.0472, 1.0472)
        theta2 = np.clip(theta2, .785398, 1.48353)
        theta3 = np.clip(theta3, 1.5708, 1.8326)
        theta4 = np.clip(theta4, 1.047202, 1.8326)

        reward, done = reward_function(state[0:3], state[3:6])

        prediction_error = state[0:3] - state[3:6]

        ball_index+=1

        next_state =  np.concatenate((ball_path[ball_index], displacement_function(theta1, theta2, theta3, theta4), prediction_error)).astype(np.float32)

        # reward, done = reward_function(next_state[:3], next_state[3:])

        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state