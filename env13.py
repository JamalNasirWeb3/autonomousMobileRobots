from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import math
import cv2
import matplotlib.pyplot as plts


class CoppeliaSimEnv:
    def __init__(self, num_obstacles=0, vision_sensor_name='/Vision_sensor', target_name='/Cone', downscale_factor=2, process_frequency=10):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.num_obstacles = num_obstacles
        self.vision_sensor_name = vision_sensor_name
        self.target_name = target_name
        self.downscale_factor = downscale_factor
        self.process_frequency = process_frequency
        self.step_count = 0
        self.previous_distance_to_goal = float('inf')
        self.left_motor_handle, self.right_motor_handle, self.robot_handle, self.obstacle_handles, self.vision_sensor_handle, self.target_handle = self.get_handles()
        self.action_space = 2  # Action space for left and right motor speeds
        self.state_space = 3 + 3 + 2 + 3 * self.num_obstacles + self.get_vision_sensor_state_space()  # State space includes position, orientation, relative target position, obstacles, and vision sensor image
        self.boundary_limits = [-2.0, 2.0, -2.0, 2.0]  # Define boundary limits
        self.obstacle_detected = False  # Flag to indicate obstacle detection
        self.avoidance_steps = 0  # Steps for obstacle avoidance
        self.turn_direction = 1  # 1 for right turn, -1 for left turn

        self.all_positions = []
        self.all_distances_to_goal = []
        self.all_distances_to_obstacles = []

        # Initialize lists to store data for plotting
        self.positions = []
        self.distances_to_goal = []
        self.distances_to_obstacles = []

    def get_handles(self):
        try:
            left_motor_handle = self.sim.getObject('/leftMotor')
            right_motor_handle = self.sim.getObject('/rightMotor')
            robot_handle = self.sim.getObject('/PioneerP3DX')
            obstacle_handles = [self.sim.getObject(f'/Cuboid[{i}]') for i in range(self.num_obstacles)]
            vision_sensor_handle = self.sim.getObject(self.vision_sensor_name)
            target_handle = self.sim.getObject(self.target_name)

            return left_motor_handle, right_motor_handle, robot_handle, obstacle_handles, vision_sensor_handle, target_handle
        except Exception as e:
            print(f"Error retrieving object handles: {e}")
            return -1, -1, -1, [], -1, -1  # Return invalid handles

    def get_vision_sensor_state_space(self):
        vision_sensor_resolution = self.sim.getVisionSensorResolution(self.vision_sensor_handle)
        downscaled_resolution = (vision_sensor_resolution[0] // self.downscale_factor) * (vision_sensor_resolution[1] // self.downscale_factor)
        return downscaled_resolution  # Assuming grayscale image

    def reset(self):
        self.sim.stopSimulation()
        time.sleep(0.5)
        self.sim.startSimulation()
        time.sleep(0.5)

        # Set fixed initial position for the robot but do not reset the target
        robot_initial_position = [-0.6,1.1,0]

        # Set the robot's position
        self.sim.setObjectPosition(self.robot_handle, -1, robot_initial_position)

        self.step_count = 0
        self.previous_distance_to_goal = float('inf')
        self.obstacle_detected = False
        self.sim.setShapeColor(self.robot_handle, None, self.sim.colorcomponent_ambient_diffuse, [1, 1, 1])  # Reset color to default
        self.avoidance_steps = 0
        self.turn_direction = 1  # Reset turn direction

        # Reset tracking lists
        self.positions = []
        self.distances_to_goal = []
        self.distances_to_obstacles = []

        return self.get_state()

    def get_state(self):
        try:
            if self.robot_handle == -1:
                raise ValueError("Robot handle is invalid.")
            
            position = self.sim.getObjectPosition(self.robot_handle, -1)
            orientation = self.sim.getObjectOrientation(self.robot_handle, -1)
            target_position = self.sim.getObjectPosition(self.target_handle, -1)
            relative_target_position = [target_position[0] - position[0], target_position[1] - position[1]]

            obstacle_positions = []
            for handle in self.obstacle_handles:
                if handle != -1:
                    obstacle_position = self.sim.getObjectPosition(handle, -1)
                else:
                    obstacle_position = [0, 0, 0]
                obstacle_positions.extend(obstacle_position)
            
            if self.step_count % self.process_frequency == 0:
                vision_sensor_image = self.sim.getVisionSensorImage(self.vision_sensor_handle)
                vision_sensor_image = np.array(vision_sensor_image, dtype=np.float32)
                vision_sensor_image = vision_sensor_image.reshape((self.sim.getVisionSensorResolution(self.vision_sensor_handle)[1], self.sim.getVisionSensorResolution(self.vision_sensor_handle)[0], 3))
                vision_sensor_image = cv2.cvtColor(vision_sensor_image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                vision_sensor_image = cv2.resize(vision_sensor_image, (vision_sensor_image.shape[1] // self.downscale_factor, vision_sensor_image.shape[0] // self.downscale_factor))  # Downscale
                self.vision_sensor_image = vision_sensor_image.flatten()  # Flatten image to vector

            state = np.concatenate((position, orientation, relative_target_position, obstacle_positions, self.vision_sensor_image))

            if len(state) != self.state_space:
                raise ValueError(f"State vector length {len(state)} does not match state_space {self.state_space}.")
        except Exception as e:
            print(f"Error getting state: {e}")
            state = np.zeros(self.state_space)
        return state

    def step(self, action):
        self.step_count += 1

        if not isinstance(action, (np.ndarray, list)):
            raise ValueError("Action must be a list or numpy array of two elements (left_speed, right_speed)")
        if len(action) != 2:
            raise ValueError("Action array size must be 2")

        max_motor_velocity = 2.0  # Define a suitable max velocity

        if self.avoidance_steps > 0:
            # If in avoidance mode, continue turning
            left_speed, right_speed = 0.5 * self.turn_direction, -0.5 * self.turn_direction
            self.avoidance_steps -= 1
        else:
            # Calculate the action using the potential field approach
            left_speed, right_speed = self.calculate_action(self.get_state())

        # Set the motor speeds
        self.set_motor_speed(left_speed * max_motor_velocity, right_speed * max_motor_velocity)
        time.sleep(0.1)

        # Get the current state, reward, and check if done
        state = self.get_state()
        reward = self.calculate_reward(state)
        done = self.check_done(state)

        # Track the robot's position and distance to goal
        x, y = state[0], state[1]
        self.positions.append((x, y))
        goal_position = self.sim.getObjectPosition(self.target_handle, -1)
        distance_to_goal = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2)
        self.distances_to_goal.append(distance_to_goal)

        # Track the closest distance to an obstacle
        distances_to_obstacles = [
            self.distance([x, y], self.sim.getObjectPosition(handle, -1)[:2])
            for handle in self.obstacle_handles
        ]
        self.distances_to_obstacles.append(min(distances_to_obstacles))

        if self.check_out_of_bounds(state):
            reward -= 10  # Apply a penalty for going out of bounds
            done = True



        # Check for obstacle collision
        if self.check_obstacle_collision():
            print("Obstacle detected! Collision with obstacle.")
            self.set_motor_speed(0, 0)  # Stop the robot
            self.sim.setShapeColor(self.robot_handle, None, self.sim.colorcomponent_ambient_diffuse, [0, 0, 1])  # Blue color
            self.avoidance_steps = 20  # Number of steps to turn right or left
            self.turn_direction = -self.turn_direction  # Change turn direction
            # Reset the simulation state
            state = self.reset()
            reward = -10  # Large negative reward for collision
            done = True

        # Check if target is reached
        if self.check_done(state):
            print("Target reached!")
            self.set_motor_speed(0, 0)  # Stop the robot
            reward = 10  # Large positive reward for reaching the goal
            done = True

        return state, reward, done, {}

    def calculate_action(self, state):
        x, y, theta = state[0], state[1], state[2]
        goal_position = self.sim.getObjectPosition(self.target_handle, -1)

        # Attractive force towards the goal
        k_attr = 3.0  # Increased attractive force gain
        f_attr_x = k_attr * (goal_position[0] - x)
        f_attr_y = k_attr * (goal_position[1] - y)

        # Repulsive force away from obstacles with lookahead
        k_rep = 0.3  # Slightly increased repulsive force gain
        f_rep_x, f_rep_y = 0.0, 0.0
        robot_position = np.array([x, y])
        lookahead_distance = 0.5  # Lookahead distance for predicting potential collisions

        for handle in self.obstacle_handles:
            obstacle_position = np.array(self.sim.getObjectPosition(handle, -1))
            distance_to_obstacle = np.linalg.norm(obstacle_position[:2] - robot_position)

            if distance_to_obstacle < 1.5:  # Only consider obstacles within a certain range
                # Predict future position considering current trajectory
                k_rep = 0.2 * (1.5 - distance_to_obstacle)
                future_position = robot_position + lookahead_distance * np.array([math.cos(theta), math.sin(theta)])
                future_distance_to_obstacle = np.linalg.norm(obstacle_position[:2] - future_position)

                # Apply stronger repulsive force if future collision is likely
                if future_distance_to_obstacle < distance_to_obstacle:
                    repulsion_magnitude = k_rep * (1.0 / future_distance_to_obstacle - 1.0) / (future_distance_to_obstacle ** 2)
                    f_rep_x += repulsion_magnitude * (future_position[0] - obstacle_position[0])
                    f_rep_y += repulsion_magnitude * (future_position[1] - obstacle_position[1])
                else:
                    repulsion_magnitude = k_rep * (1.0 / distance_to_obstacle - 1.0) / (distance_to_obstacle ** 2)
                    f_rep_x += repulsion_magnitude * (robot_position[0] - obstacle_position[0])
                    f_rep_y += repulsion_magnitude * (robot_position[1] - obstacle_position[1])

        # Combine attractive and repulsive forces
        f_total_x = f_attr_x + f_rep_x
        f_total_y = f_attr_y + f_rep_y
        angle_to_move = math.atan2(f_total_y, f_total_x)

        # Calculate motor speeds based on the angle to move
       # Refine motor speedsstate
        desired_velocity = 1.0
        angular_correction = np.clip(angle_to_move - theta, -0.5, 0.5)
        left_speed = desired_velocity - angular_correction
        right_speed = desired_velocity + angular_correction

        # Normalize speeds to maintain them within the valid range
        left_speed = np.clip(left_speed, -1, 1)
        right_speed = np.clip(right_speed, -1, 1)

        print(f"Attractive Force: ({f_attr_x}, {f_attr_y})")
        print(f"Repulsive Force: ({f_rep_x}, {f_rep_y})")
        # print(f"Total Force: ({f_total_x}, f_total_y})")
        # print(f"Angle to Move: {angle_to_move}")
        # print(f"Angular Correction: {angular_correction}")

        return [left_speed, right_speed]

    def set_motor_speed(self, left_speed, right_speed):
        try:
            print(f"Setting motor speeds: Left={left_speed}, Right={right_speed}")
            self.sim.setJointTargetVelocity(self.left_motor_handle, float(left_speed))
            self.sim.setJointTargetVelocity(self.right_motor_handle, float(right_speed))
        except Exception as e:
            print(f"Error setting motor speed: {e}")

    def calculate_reward(self, state):
        x, y, theta = state[0], state[1], state[2]
        goal_position = self.sim.getObjectPosition(self.target_handle, -1)  # Get target position
        distance_to_goal = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2)
        print(f"Distance to goal: {distance_to_goal}")

        if self.check_obstacle_collision() or self.check_out_of_bounds(state):
            reward = -10  # Large negative reward for collision or going out of bounds
        elif distance_to_goal < 0.1:
            reward = 10  # Large positive reward for reaching the goal
        else:
            reward = -distance_to_goal  # Negative reward based on distance to the target
            if distance_to_goal < self.previous_distance_to_goal:
                reward += 1.0  # Positive reward for moving closer to the goal
            else:
                reward -= 2.0  # Negative reward for moving away from the goal

        self.previous_distance_to_goal = distance_to_goal

        return reward

    # def check_out_of_bounds(self, state):
    #     x, y = state[0], state[1]
    #     x_min, x_max, y_min, y_max = self.boundary_limits
    #     if not (x_min < x < x_max and y_min < y < y_max):
    #         print(f"Robot is out of bounds: ({x}, {y})")
    #         return True
    #     return False

    def check_obstacle_collision(self):
        try:
            robot_position = self.sim.getObjectPosition(self.robot_handle, -1)
            distance_threshold = 0.3  # Increased distance threshold for early avoidance

            for handle in self.obstacle_handles:
                obstacle_position = self.sim.getObjectPosition(handle, -1)
                distance = self.distance(robot_position[:2], obstacle_position[:2])  # Compare only x and y dimensions
                print(f"Distance to obstacle: {distance}")
                if distance < distance_threshold:
                    self.obstacle_detected = True
                    print("Collision detected! Adjusting path...")
                    return True  # Collision detected
        except Exception as e:
            print(f"Error checking for obstacle collision: {e}")

        self.obstacle_detected = False
        return False

    def check_out_of_bounds(self, state):
        x, y = state[0], state[1]
        x_min, x_max, y_min, y_max = self.boundary_limits
        if not (x_min < x < x_max and y_min < y < y_max):
            print(f"Robot is out of bounds: ({x}, {y})")
            return True
        return False

    def check_done(self, state):
        x, y = state[0], state[1]
        goal_position = self.sim.getObjectPosition(self.target_handle, -1)  # Get target position
        distance_to_goal = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2)

        print(f"Robot Position: ({x}, {y})")
        print(f"Target Position: {goal_position}")
        print(f"Distance to Goal: {distance_to_goal}")
        print(f"Out of Bounds Check: {self.check_out_of_bounds(state)}")
    
        # Ensure that the robot is within bounds before checking if it has reached the target
        if self.check_out_of_bounds(state):
            print(f"Robot is out of bounds: ({x}, {y})")
            return False  # Do not declare the target as reached if out of bounds

        # Now check if the robot has reached the target
        if distance_to_goal < 0.1:
            print("Target reached!")
            return True  # Done if close to goal

        if self.check_obstacle_collision():
            print("Collision detected! Adjusting path...")
            return False  # Do not end the episode; handle collision recovery
    
        return False

    def distance(self, point1, point2):
        # Calculate distance considering only the first two dimensions (x, y)
        return np.linalg.norm(np.array(point1)[:2] - np.array(point2)[:2])

    def accumulate_episode_data(self):
        """Accumulate data at the end of each episode."""
        self.all_positions.append(self.positions)
        self.all_distances_to_goal.append(self.distances_to_goal)
        self.all_distances_to_obstacles.append(self.distances_to_obstacles)
    
    def plot_cumulative_behaviors(self):
    
        print("Plotting cumulative robot behavior...")

    # Plot the cumulative robot path
        plts.figure(figsize=(10, 5))
        for episode, episode_positions in enumerate(self.all_positions):
            episode_positions = np.array(episode_positions)
            plts.plot(episode_positions[:, 0], episode_positions[:, 1], label=f'Episode {episode + 1}')
        plts.scatter([self.sim.getObjectPosition(self.target_handle, -1)[0]], [self.sim.getObjectPosition(self.target_handle, -1)[1]], c='red', label='Target')
        plts.xlabel('X Position')
        plts.ylabel('Y Position')
        plts.title('Cumulative Robot Path Across All Episodes')
        plts.legend()
        plts.show()

        #plts.figure(figsize=(10, 5))
        print("helo")
        plts.figure(figsize=(10, 5))
        for episode, episode_distances in enumerate(self.all_distances_to_goal):
            plts.plot(episode_distances, label=f'Episode {episode + 1}')
        plts.xlabel('Step')
        plts.ylabel('Distance to Goal')
        plts.title('Cumulative Distance to Goal Over Time Across All Episodes')
        plts.legend()
        plts.show()



    def plot_robot_behavior(self):
        # Plot the robot's path for all episodes
        plt.figure(figsize=(10, 5))
        for episode, positions in enumerate(self.all_positions):
            positions = np.array(positions)
            plt.plot(positions[:, 0], positions[:, 1], label=f'Episode {episode + 1}')
        plt.scatter([self.sim.getObjectPosition(self.target_handle, -1)[0]], [self.sim.getObjectPosition(self.target_handle, -1)[1]], c='red', label='Target')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Robot Path Across Episodes')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot the distance to the goal over time for all episodes
        plt.figure(figsize=(10, 5))
        for episode, distances in enumerate(self.all_distances_to_goal):
            plt.plot(distances, label=f'Episode {episode + 1}')
        plt.xlabel('Step')
        plt.ylabel('Distance to Goal')
        plt.title('Distance to Goal Over Time Across Episodes')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot the distance to the closest obstacle over time for all episodes
        plt.figure(figsize=(10, 5))
        for episode, distances in enumerate(self.all_distances_to_obstacles):
            plt.plot(distances, label=f'Episode {episode + 1}')
        plt.xlabel('Step')
        plt.ylabel('Distance to Closest Obstacle')
        plt.title('Distance to Closest Obstacle Over Time Across Episodes')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_distance_graph(self):
        plts.figure(figsize=(10, 5))
    
        for episode, (distances_to_goal, distances_to_obstacles) in enumerate(zip(self.all_distances_to_goal, self.all_distances_to_obstacles)):
            plts.plot(distances_to_goal, label=f"Episode {episode + 1} - Distance to Target", color='blue', alpha=0.5)
            plts.plot(distances_to_obstacles, label=f"Episode {episode + 1} - Distance to Nearest Obstacle", color='red', alpha=0.5)
    
        plts.xlabel('Step')
        plts.ylabel('Distance')
        plts.title('Distance to Target and Nearest Obstacle Over Time Across All Episodes')
        plts.legend()
        plts.grid(True)
        plts.show()
        


    def close(self):
        try:
            self.sim.stopSimulation()
            time.sleep(0.5)
        except Exception as e:
            print(f"Error stopping simulation: {e}")
