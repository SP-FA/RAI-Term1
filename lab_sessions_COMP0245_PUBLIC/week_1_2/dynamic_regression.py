from statistics import covariance

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)
        tau_mes_all.append(tau_mes)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # TODO Compute regressor and store it
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        regressor_all.append(cur_regressor)
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    tau_mes_all = np.array(tau_mes_all)  # u(x), (n, j)
    regressor_all = np.array(regressor_all)  # Y(g, g_d, g_dd), (n, j, p)
    n, j, p = regressor_all.shape

    XT = regressor_all.transpose(1, 2, 0)
    X  = regressor_all.transpose(1, 0, 2)  # (j, n, p)
    y = tau_mes_all[:, :, np.newaxis].transpose(1, 0, 2)  # (j, n, 1)

    regressor_inv = np.linalg.pinv(X)  # (j, p, n)
    a = regressor_inv @ y  # (j, p, 1)
    u_hat = X @ a  # (j, n, 1)

    a = np.squeeze(a, axis=-1)  # (j, p)
    print(f"a: {a}")
    print(f"a_shape: {a.shape}")
    # print(a[:, :, np.newaxis].transpose(1, 0, 2) - beta_hat)

    # TODO compute the metrics for the linear model
    RSS = np.sum((y - u_hat) ** 2, axis=1) + 1e-9  # (j, 1)
    TSS = np.sum((y - np.mean(y, axis=1)[:, :, np.newaxis]) ** 2, axis=1)# + 1e-9

    r2 = 1 - RSS / TSS
    r2_adj = 1 - ((1  - r2) * (n - 1) / (n - j - 1))  # (j, 1)
    print(f"r2_adj: {r2_adj}")

    F = (TSS - RSS) * (n - j - 1) / (RSS * j)  # (j, 1)
    print(f"F: {F}")

    sigma2 = RSS / (n - j - 1)  # (j, 1)
    ex_sigma2 = sigma2[:, np.newaxis]
    covariance = (XT @ X) * ex_sigma2  # (j, p, p)
    se_beta = np.sqrt(np.diagonal(np.linalg.pinv(covariance) + 1e-9, axis1=1, axis2=2))
    interval_low  = a - 1.96 * se_beta
    interval_high = a + 1.96 * se_beta
    print(f"params interval low: {interval_low}")
    print(f"params interval high: {interval_high}")
    print(f"params interval low / high shape: {interval_low.shape}")

    # print(f"se_beta: {se_beta}")
    # print(f"se_beta_shape: {se_beta.shape}")

    se_pred = np.sqrt(np.diagonal(X @ np.linalg.pinv(XT @ X) @ XT + 1, axis1=1, axis2=2) * sigma2)
    u_hat = np.squeeze(u_hat, axis=-1)
    interval_low = u_hat - 1.96 * se_pred
    interval_high = u_hat + 1.96 * se_pred
    print(f"pred interval low: {interval_low}")
    print(f"pred interval high: {interval_high}")
    print(f"pred interval low / high shape: {interval_low.shape}")
    # print(se_pred)
    # print(se_pred.shape)

    # TODO plot the torque prediction error for each joint (optional)
    MSE = RSS / n
    plt.figure()
    plt.plot(MSE, "b", label="q")
    plt.title("R2")
    plt.xlabel("joint id")
    plt.ylabel("R2")
    plt.grid(True)
    plt.show()
    

if __name__ == '__main__':
    main()
