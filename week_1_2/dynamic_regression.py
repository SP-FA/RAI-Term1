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
    tau_cmd_all = []
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
        tau_cmd_all.append(feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd))

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
    # tau_cmd_all: u(x), (n, p)
    # tau_mes_all: u^(x), (n, p)
    # regressor_all: Y(g, g_d, g_dd)  (n, p, 70)
    n = 10001
    p = 7
    a_all = []
    for i in range(n):
        cur_reg = regressor_all[i]  # (p, 70)
        cur_tau = tau_mes_all[i]  # (p)
        a = np.linalg.pinv(cur_reg) @ cur_tau
        a_all.append(a)
    
    # TODO compute the metrics for the linear model
    tau_cmd_all = np.array(tau_cmd_all)
    tau_mes_all = np.array(tau_mes_all)
    RSS = np.sum((tau_cmd_all - tau_mes_all) ** 2, axis=1)
    TSS = np.sum(tau_cmd_all - np.expand_dims(np.mean(tau_mes_all, axis=1), axis=1), axis=1)
    
    r2 = 1 - RSS / TSS
    r2_adj = 1 - ((1  - r2) * (n - 1) / (n - p - 1))
    print(f"r2_adj: {r2_adj}")
    print(r2_adj.shape)
    
    F = ((TSS - RSS) / p) / (RSS / (n - p - 1))
    print(f"F: {F}")
    print(F.shape)

    SE_pred_all = []
    for i in range(n):
        X = regressor_all[i]
        kernel = np.linalg.pinv(np.dot(X.T, X))
        s = np.sum(np.dot(X, kernel) * X, axis=1)
        SE_pred = np.sqrt(np.var(X, axis=1) * (1 + s))
        SE_pred_all.append(SE_pred)

    SE_pred_all = np.array(SE_pred_all)
    interval_low = tau_mes_all - 1.96 * SE_pred_all
    interval_high = tau_mes_all + 1.96 * SE_pred_all
    print(interval_low)
    print(interval_high)
   
    # TODO plot the  torque prediction error for each joint (optional)
    

if __name__ == '__main__':
    main()
