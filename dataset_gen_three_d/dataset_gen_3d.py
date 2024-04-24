import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

# Random seed for the training data is 225
# random seed for the test data is 522
random.seed(522)

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

#if the position is outside of the effective range of the arm, quit recording the trajectory
def within_range(Xpos, Ypos, x_max, y_max):
    #euc_dis_from_origin = math.sqrt((Xpos**2) + (Ypos**2) + (Zpos**2))

    #if euc_dis_from_origin > radius:
    if abs(Xpos) > x_max or abs(Ypos) > y_max:
        return False
    else:
        return True
    
sample_generation_number = 100
x_vals, y_vals, z_vals, done_flag = [],[],[],[]

elapsed_time = 0

for i in range(sample_generation_number):

    area_prob = random.uniform(0,1)

    x_pos_sample, y_pos_sample, z_pos_sample, done_sample = [],[],[],[]

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

    # I think the error in the trajectory might just be due to the inclusion of the angles. need to delete those and see how it changes

    psi = np.float32(np.radians(random.uniform(1,89)))  # angle relative to x plane (radians)
    theta = np.float32(np.radians(random.uniform(1,89))) # angle relative to y plane (radians)
    phi = np.float32(np.radians(random.uniform(1,89))) # angle relative to z plane (radians)

    initial_x_velocity = np.float32(initial_velocity * np.cos(psi)) # meters/second
    initial_y_velocity = np.float32(initial_velocity * np.cos(theta)) # meters/second
    initial_z_velocity = np.float32(initial_velocity * np.cos(phi)) # meters/second

    # shift the x position over, so that the tip of the triangle is the origin of the frame, then convert the centimeters to meters
    x_position = np.float32((x_rand - (x_max/2))/100)

    # convert the centimeters to meters for the y position
    y_position = np.float32(y_rand/100)

    # the ball starts on the ground, so z = 0
    z_position = np.float32(0)

    x_pos_sample.append(x_position)
    y_pos_sample.append(y_position)
    z_pos_sample.append(z_position)
    done_sample.append(True)


    # Lists to store the trajectory
    #x_vals, y_vals, z_vals = [x_position], [y_position], [z_position]
    x_velocity, y_velocity, z_velocity = initial_x_velocity, initial_y_velocity, initial_z_velocity

    time_steps_sample = 1

    # Simulation loop for a single sample (one throw of the ball)
    # the average number of time steps tends to be around 28, letting it run for 30 times steps allows it to be a nice number
    # this gives the agent about half a second maximum to catch the ball once it enters within its range
    #while z_position >= 0  and time_steps_sample < 30 and within_range(x_position, y_position, z_position, sphere_radius):
    while z_position >= 0 and within_range(x_position, y_position, x_max, y_max):
        
        elapsed_time+=1

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
        done_sample.append(False)
    
    x_pos_sample.reverse()
    y_pos_sample.reverse()
    z_pos_sample.reverse()
    done_sample.reverse()


    x_vals.append(np.array(x_pos_sample, dtype=object))
    y_vals.append(np.array(y_pos_sample, dtype=object))
    z_vals.append(np.array(z_pos_sample, dtype=object))
    done_flag.append(np.array(done_sample, dtype=object))

x_vals_flat = np.concatenate(x_vals)
y_vals_flat = np.concatenate(y_vals)
z_vals_flat = np.concatenate(z_vals)
done_flag_flat = np.concatenate(done_flag)

data_types = {
    'Xpos': 'float32',
    'Ypos': 'float32',
    'Zpos': 'float32',
    'DoneFlag': 'bool'
}

df = pd.DataFrame(
    {
        'Xpos': x_vals_flat,
        'Ypos': y_vals_flat,
        'Zpos': z_vals_flat, 
        'DoneFlag': done_flag_flat 
    }
)

# change the types to save space
df = df.astype(data_types)

# df to parquet table
pa_table = pa.Table.from_pandas(df)

# Write Arrow Table to Parquet file
pq.write_table(pa_table, 'model_test_data_{}.parquet'.format(sample_generation_number))