import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models

# Set the model type: "neural_network" or "random_forest"
neural_network_or_random_forest = "random_forest"  # Change to "random_forest" to use Random Forest models

max_depth = 2

# Task 3.3
def exponential_moving_average(data, alpha):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

def double_exponential_smoothing(data, alpha, beta):
    level, trend = data[0], data[1] - data[0]
    result = [level]
    for t in range(1, len(data)):
        last_level = level
        level = alpha * data[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level)
    return result

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(128, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)

def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
        return
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])            # Shape: (N,)
        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Load all the models in a list
    models = []
    if neural_network_or_random_forest == "neural_network":
        for joint_idx in range(7):
            # Instantiate the model
            model = MLP()
            # Load the saved model
            model_filename = os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            models.append(model)
    elif neural_network_or_random_forest == "random_forest":
        for joint_idx in range(7):
            # Load the saved Random Forest model
            model_filename = os.path.join(script_dir, f'rf_joint{max_depth}_{joint_idx+1}.joblib')
            model = joblib.load(model_filename)
            models.append(model)
    else:
        print("Invalid model type specified. Please set neural_network_or_random_forest to 'neural_network' or 'random_forest'")
        return

    # Generate a new goal position
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    # Create a set of goal positions
    number_of_goal_positions_to_test = 10
    goal_positions = []
    for i in range(number_of_goal_positions_to_test):
        goal_positions.append([
            np.random.uniform(*goal_position_bounds['x']),
            np.random.uniform(*goal_position_bounds['y']),
            np.random.uniform(*goal_position_bounds['z'])
        ])

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # PD controller gains
    kp = 1000  # Proportional gain
    kd = 100   # Derivative gain

    # Get joint velocity limits
    joint_vel_limits = sim.GetBotJointsVelLimit()

    time_step = sim.GetTimeStep()
    # Generate test time array
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    q_mes_res = []
    q_des_res = []
    q_d_mes_res = []
    tau_cmd_res = []
    mse_res = 0

    for goal_position in goal_positions:
        print("Testing new goal position------------------------------------")
        print(f"Goal position: {goal_position}")

        q_mes_all = []
        q_des_all = []
        q_d_mes_all = []
        tau_cmd_all = []

        # Initialize the simulation
        sim.ResetPose()
        current_time = 0  # Initialize current time

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (num_points, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (num_points, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)

        for joint_idx in range(7):
            if neural_network_or_random_forest == "neural_network":
                # Prepare the test input
                test_input_tensor = torch.from_numpy(test_input).float()  # Shape: (num_points, 4)

                # Predict joint positions using the neural network
                with torch.no_grad():
                    predictions = models[joint_idx](test_input_tensor).numpy().flatten()  # Shape: (num_points,)
            elif neural_network_or_random_forest == "random_forest":
                # Predict joint positions using the Random Forest
                predictions = models[joint_idx].predict(test_input)  # Shape: (num_points,)
                # Task 3.3
                predictions = exponential_moving_average(predictions, 0.001)
                # predictions = double_exponential_smoothing(predictions, 0.001, 0.01)

            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = predictions

        # Compute qd_des_over_time by numerically differentiating the predicted joint positions
        qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
        # Clip the joint velocities to the joint limits
        qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))

        # Data collection loop
        while current_time < test_time_array.max():
            # Measure current state
            q_mes = sim.GetMotorAngles(0)  # (7,)
            q_mes_all.append(q_mes)

            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            q_d_mes_all.append(qd_mes)

            # Get the index corresponding to the current time
            current_index = int(current_time / time_step)
            if current_index >= len(test_time_array):
                current_index = len(test_time_array) - 1

            # Get q_des and qd_des_clip from predicted data
            q_des = predicted_joint_positions_over_time[current_index, :]  # Shape: (7,)
            qd_des_clip = qd_des_over_time_clipped[current_index, :]      # Shape: (7,)
            q_des_all.append(q_des)

            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            tau_cmd_all.append(tau_cmd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command

            # Keyboard event handling
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            # Exit logic with 'q' key
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            # Time management
            time.sleep(time_step)  # Control loop timing
            current_time += time_step

        # After the trajectory, compute the final cartesian position
        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
        print(f"Final computed cartesian position: {final_cartesian_pos}")
        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")

        q_mes_res.append(q_mes_all)
        q_des_res.append(q_des_all)
        q_d_mes_res.append(q_d_mes_all)
        tau_cmd_res.append(tau_cmd_all)
        mse_res += position_error ** 2

    q_mes_res = np.array(q_mes_res)
    q_des_all = np.array(q_des_res)
    q_d_mes_res = np.array(q_d_mes_res)
    tau_cmd_res = np.array(tau_cmd_res)
    return q_mes_res, q_des_all, q_d_mes_res, tau_cmd_res, mse_res

if __name__ == '__main__':
    neural_network_or_random_forest = "random_forest"
    rf_q_mes_1, rf_q_des_1, rf_qd_mes_1, rf_t_cmd_1, rf_mse_1 = main()
    neural_network_or_random_forest = "neural_network"
    nn_q_mes, nn_q_des, nn_qd_mes, nn_t_cmd, nn_mse = main()
    max_depth = 10
    neural_network_or_random_forest = "random_forest"
    rf_q_mes_2, rf_q_des_2, rf_qd_mes_2, rf_t_cmd_2, rf_mse_2 = main()

    print(f"nn mse: {nn_mse}")
    print(f"rf mse 1: {rf_mse_1}")
    print(f"rf mse 2: {rf_mse_2}")

    # Task 3.1:
    # for i in range(7):
    #     plt.figure()
    #     for j in range(10):
    #         nn = nn_q_mes[j, :, i]
    #         rf_1 = rf_q_mes_1[j, :, i]
    #         rf_2 = rf_q_mes_2[j, :, i]
    #
    #         plt.plot(nn, color="blue", alpha=0.3)
    #         plt.plot(rf_1, color="red", alpha=0.3)
    #         plt.plot(rf_2, color="green", alpha=0.3)
    #         plt.xlabel("Time")
    #         plt.ylabel("Position")
    #         plt.grid(True)
    #     plt.show()
    #
    # for i in range(7):
    #     plt.figure()
    #     for j in range(10):
    #         nn = nn_qd_mes[j, :, i]
    #         rf_1 = rf_qd_mes_1[j, :, i]
    #         rf_2 = rf_qd_mes_2[j, :, i]
    #
    #         plt.plot(nn, color="blue", alpha=0.3)
    #         plt.plot(rf_1, color="red", alpha=0.3)
    #         plt.plot(rf_2, color="green", alpha=0.3)
    #         plt.xlabel("Time")
    #         plt.ylabel("Velocity")
    #         plt.grid(True)
    #     plt.show()

    # Task 3.2:
    # for i in range(7):
    #     plt.figure()
    #     for j in range(10):
    #         nn = nn_q_des[j, :, i] - nn_q_mes[j, :, i]
    #         rf_1 = rf_q_des_1[j, :, i] - rf_q_mes_1[j, :, i]
    #         rf_2 = rf_q_des_2[j, :, i] - rf_q_mes_2[j, :, i]
    #
    #         nn[0] = 0
    #         rf_1[0] = 0
    #         rf_2[0] = 0
    #
    #         plt.plot(nn, color="blue", alpha=0.3)
    #         plt.plot(rf_1, color="red", alpha=0.3)
    #         plt.plot(rf_2, color="green", alpha=0.3)
    #         plt.xlabel("Time")
    #         plt.ylabel("Error")
    #         plt.grid(True)
    #     plt.show()
    #
    # for i in range(7):
    #     plt.figure()
    #     for j in range(10):
    #         nn = nn_t_cmd[j, :, i]
    #         rf_1 = rf_t_cmd_1[j, :, i]
    #         rf_2 = rf_t_cmd_2[j, :, i]
    #
    #         nn[0] = 0
    #         rf_1[0] = 0
    #         rf_2[0] = 0
    #
    #         plt.plot(nn, color="blue", alpha=0.3)
    #         plt.plot(rf_1, color="red", alpha=0.3)
    #         plt.plot(rf_2, color="green", alpha=0.3)
    #         plt.xlabel("Time")
    #         plt.ylabel("Torque")
    #         plt.grid(True)
    #     plt.show()

    # Task 3.3:
    # for i in range(7):
    #     plt.figure()
    #     for j in range(10):
    #         nn = nn_q_des[j, :, i] - nn_q_mes[j, :, i]
    #         rf_1 = rf_q_des_1[j, :, i] - rf_q_mes_1[j, :, i]
    #         rf_2 = rf_q_des_2[j, :, i] - rf_q_mes_2[j, :, i]
    #
    #         nn[0] = 0
    #         rf_1[0] = 0
    #         rf_2[0] = 0
    #
    #         plt.plot(nn, color="blue", alpha=0.3)
    #         plt.plot(rf_1, color="red", alpha=0.3)
    #         plt.plot(rf_2, color="green", alpha=0.3)
    #         plt.xlabel("Time")
    #         plt.ylabel("Error")
    #         plt.grid(True)
    #     plt.show()
