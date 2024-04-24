import math
import random
import matplotlib.pyplot as plt
import numpy as np

#This code is basically functional, need to fix one error in the output, otherwise finished

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
    


# define the initial starting params (triangular bounds don't change)
# param_intialize( theta_one(deg) , theta_two(deg), side_length(centimeters))
sample_generation_number = 250000
x_min, x_max, y_min, y_max, left_slope, left_intercept, right_slope, right_intercept = param_initialize(30, 150, 70.64)

triangle_area = y_max * x_max * 0.5
semi_circ_area = math.pi * ((x_max/2) ** 2) * 0.5 
total_area = triangle_area + semi_circ_area
semi_circ_prob = semi_circ_area / total_area

point_dict = {}

for i in range(sample_generation_number):

    area_prob = random.uniform(0,1)

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
            
        
        point_dict["{:.2f}".format(x_rand)] = "{:.2f}".format(y_rand)
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
        
        point_dict["{:.2f}".format(x_rand)] = "{:.2f}".format(y_rand)
    



# shifts over all the x points, so that the center of the tip of the triangle is centered on x=0 , since that's the origin of the frame
# divides both positions by 100 to convert from centimeters to meters
x_key_float = [(float(x) - (x_max/2))/100 for x in point_dict.keys()]
y_val_float = [(float(y))/100 for y in point_dict.values()]

#points = [(x,y) for x,y in zip(x_key_float,y_val_float)]

# plt.hexbin(x_key_float, y_val_float, gridsize=25, cmap='hot', mincnt=1)
# plt.colorbar(label='Frequency')
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.title('Heatmap of XY point plotting')
# plt.grid(True)
# plt.show()

plt.scatter(x_key_float, y_val_float)
#plt.imshow(np.asarray(points), cmap='hot', interpolation='nearest')
plt.xlabel('X coordinate (meters)')
plt.ylabel('Y coordinate (meters)')
plt.title('Mapping of random (X,Y) point generation within working bounds')
plt.show()