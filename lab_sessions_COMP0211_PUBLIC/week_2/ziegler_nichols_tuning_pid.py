import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")



# single joint tuning
#episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd = np.array([0]*dyn_model.getNumberofActuatedJoints())
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement 

   
    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors


    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    

    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print("current time in seconds",current_time)

    
    # TODO make the plot for the current joint
    q_mes_all = np.array(q_mes_all)
    q_mes_all = q_mes_all[:, joints_id]

    qd_mes_all = np.array(qd_mes_all)
    qd_mes_all = qd_mes_all[:, joints_id]

    qd_mes_all1 = qd_mes_all[:-1]
    qd_mes_all2 = qd_mes_all[1:]
    mask1 = np.logical_and(qd_mes_all1 <= 0, qd_mes_all2 > 0)
    mask2 = np.logical_and(qd_mes_all1 >= 0, qd_mes_all2 < 0)
    q_mes_mask = np.logical_or(mask1, mask2)

    q_mes_all_ = q_mes_all[:-1]
    q_mes_amplitude = q_mes_all_[q_mes_mask]
    q_mes_amplitude = np.abs(q_mes_amplitude - 1)
    var = np.var(q_mes_amplitude)
    if plot is True:
        plt.figure()
        # plt.plot(qd_mes_all, "r", label="qd")
        plt.plot(q_mes_all, "b", label="q")
        plt.title("Oscillation")
        plt.xlabel("t")
        plt.ylabel("c(t)")
        plt.grid(True)
        plt.show()
    
    return q_mes_all, var
     



def perform_frequency_analysis(data, dt, plot=False):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Optional: Plot the spectrum
    if plot is True:
        plt.figure()
        plt.plot(xf, power)
        plt.title("FFT of the signal")
        plt.xlabel("Frequency in Hz")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return xf, power


# TODO Implement the table in thi function
def thi_fuc(Ku, freq, power):
    power = power[1:]
    freq = freq[1:]
    dominant_freq = freq[np.argmax(power)]
    Tu = 1 / dominant_freq
    Kd = 0.1 * Ku * Tu
    Td = 0.125 * Tu
    Kp = 0.8 * Ku
    print(f"Ku: {Ku}, Tu: {Tu}, Kd: {Kd}, Td: {Td}, Kp: {Kp}")



if __name__ == '__main__':
    joint_id = 0  # Joint ID to tune
    regulation_displacement = 1  # Displacement from the initial joint position
    init_gain=1
    gain_step=1.5 
    max_gain=30
    test_duration=20 # in seconds
    

    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols methodplt.figure()
    # 0: 16    2: 13.5  3: 11  4: 16.7  5: 16.3  6: 16.6
    var_all = []
    gain_all = []
    freq_all = []
    power_all = []
    while init_gain < max_gain:
        res, var = simulate_with_given_pid_values(sim, init_gain, joint_id, regulation_displacement, test_duration, False)
        var_all.append(var)
        gain_all.append(init_gain)
        freq, power = perform_frequency_analysis(res, sim.GetTimeStep(), False)
        freq_all.append(freq)
        power_all.append(power)
        init_gain += gain_step

    idx = np.argmin(var_all)
    thi_fuc(gain_all[idx], freq_all[idx], power_all[idx])
   