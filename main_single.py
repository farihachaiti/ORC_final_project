#!/usr/bin/env python3
from utils.robot_loaders import loadUR, loadPendulum, loadURlab
from example_robot_data.robots_loader import load
from utils.robot_wrapper import RobotWrapper
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import casadi as cs
import time
from time import sleep
from adam.casadi.computations import KinDynComputations
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
from utils.viz_utils import addViewerSphere, applyViewerConfiguration
from neural_network import NeuralNetwork
import optimal_control.casadi_adam.conf_ur5 as conf_ur5
import torch
import torch.optim as optim
import torch.nn as nn

# ====================== Robot and Dynamics Setup ======================
np.set_printoptions(precision=3, linewidth=200, suppress=True)
'''r = load('double_pendulum')
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
nq, nv = robot.nq, robot.nv'''
nq = 1
nx = 2 * nq

dt = 0.2
N = 100
# Initial state (random but within reasonable bounds)
x_init = np.zeros(nx)
x_init[:nq] = (np.random.rand(nq) - 0.5) * 0.5  # Small initial angles
x_init[nq:] = 10.0  # Start from rest
q_des = np.zeros(nq)
w_p = 1.0              # position weight
w_v = 1e-2             # velocity weight
w_a = 1e-4             # acceleration weight
w_p_final = 10.0        # position weight for terminal cost
w_v_final = 1.0  

# CasADi symbolic variables
q = cs.SX.sym('q', nq)
dq = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)

m = 50.0
g = 9.81
I = 15.0

state = cs.vertcat(q, dq)
rhs = cs.vertcat(dq, ddq)
f = cs.Function('f', [state, ddq], [rhs])

q0 = np.zeros(nq)
dq0 = np.zeros(nq)
r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
simu = RobotSimulator(conf_ur5, r)
simu.init(q0, dq0)
simu.display(q0)
# Inverse dynamics
'''kinDyn = KinDynComputations(robot.urdf, joints_name_list)
H_b = cs.SX.eye(4)
v_b = cs.SX.zeros(6)
bias_forces = kinDyn.bias_force_fun()
mass_matrix = kinDyn.mass_matrix_fun()
h = bias_forces(H_b, q, v_b, dq)[6:]
M = mass_matrix(H_b, q)[6:, 6:]
tau = M @ ddq + h
inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

# Forward kinematics
fk_fun = kinDyn.forward_kinematics_fun(frame_name)
ee_pos = fk_fun(H_b, q)[:3, 3]
fk = cs.Function('fk', [q], [ee_pos])

y = fk(q0)
c_path = np.array([y[0]-r_path, y[1], y[2]]).squeeze()

lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()'''



# ====================== Optimization Problem ======================
def create_decision_variables(N, nx, nu, M=None):
    opti = cs.Opti()
    X, U = [], []
    if M is not None:
        for k in range(M+1): 
            X += [opti.variable(nx)]
        for k in range(M): 
            U += [opti.variable(nu)]
    else:
        for k in range(N+1): 
            X += [opti.variable(nx)]
        for k in range(N): 
            U += [opti.variable(nu)]
    return opti, X, U


def define_running_cost_and_dynamics(opti, X, U, N, dt, x_init, w_p, w_v, w_a, M=None):
    opti.subject_to(X[0] == x_init)
    cost = 0.0
    if M is not None:
        for k in range(M):     
            # Compute cost function
            cost += w_p * (X[k][:nq] - q_des).T @ (X[k][:nq] - q_des) * dt 
            cost += w_v * X[k][nq:].T @ X[k][nq:] * dt
            cost += w_a * U[k].T @ U[k] * dt

            # Add dynamics constraints
            opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))
            
            # Add joint position limits (adjust these values as needed)
            #opti.subject_to(opti.bounded(-np.pi, X[k][:nq], np.pi))
            
            # Add joint velocity limits (adjust these values as needed)
            max_velocity = 5.0  # rad/s
            #opti.subject_to(opti.bounded(-max_velocity, X[k][nq:], max_velocity))
    else:
        for k in range(N):     
            # Compute cost function
            cost += w_p * (X[k][:nq] - q_des).T @ (X[k][:nq] - q_des) * dt 
            cost += w_v * X[k][nq:].T @ X[k][nq:] * dt
            cost += w_a * U[k].T @ U[k] * dt

            # Add dynamics constraints
            opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))
            
            # Add joint position limits (adjust these values as needed)
            #opti.subject_to(opti.bounded(-np.pi, X[k][:nq], np.pi))
            
            # Add joint velocity limits (adjust these values as needed)
            max_velocity = 5.0  # rad/s
            #opti.subject_to(opti.bounded(-max_velocity, X[k][nq:], max_velocity))
            
    return cost


def define_terminal_cost_and_constraints(opti, X, q_des, w_p_final):
    
    #cost = w_p_final * (X[-1][:nq]).T @ (X[-1][:nq])
    cost = 0
    return cost


def create_and_solve_ocp(N, nx, nq, dt, x_init,
                         w_p, w_v, w_a, w_p_final, J_terminal=None, M=None):
    opti, X, U = create_decision_variables(N, nx, nq, M)
    running_cost = define_running_cost_and_dynamics(opti, X, U, N, dt, x_init, w_p, w_v, w_a, M)
    terminal_cost = define_terminal_cost_and_constraints(opti, X, q_des, w_p_final)
    if J_terminal is not None:
        terminal_cost += J_terminal
    opti.minimize(running_cost + terminal_cost)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    print(f"Solver time: {time.time() - t0:.2f}s")
    return sol, X, U




def extract_solution(sol, X, U):
    x_sol = np.array([sol.value(X[k]) for k in range(N + 1)]).T
    ddq_sol = np.array([sol.value(U[k]) for k in range(N)]).T
    q_sol = x_sol[:nq, :]
    dq_sol = x_sol[nq:, :]
    tau = np.zeros((nq, N))
    for i in range(N):
        tau[:, i] = I * ddq_sol[:, i] - m * g * cs.sin(q_sol[:, i]).toarray().squeeze()
    return q_sol, dq_sol, ddq_sol, tau


def display_motion(q_traj):
    for i in range(N + 1):
        t0 = time.time()
        simu.display(q_traj[:, i])
        t1 = time.time()
        if(t1-t0 < dt):
            sleep(dt - (t1-t0))



if __name__=='__main__':
    #plot_infinity(0, 1)

    log_w_v, log_w_a, log_w_final = -3, -3, 2
    log_w_p = 2 #Log of trajectory tracking cost 
    J_X_init = []
    U_init = []
    X_init_val = []
    for i in range(10):
        sol, X, U = create_and_solve_ocp(
            N, nx, nq, dt, x_init,
            log_w_p, 10**log_w_v, 10**log_w_a, 10**log_w_final, None, None)
        
        # Extract the solution values
        x_sol = np.array([sol.value(X[k]) for k in range(N + 1)]).T  # Shape: (nx, N+1)
        J_val = sol.stats()['iterations']['obj']  # Get the objective value
        
        J_X_init.append(J_val)
        X_init_val.append(x_sol)
        U_init.append(np.array([sol.value(U[k]) for k in range(N)]).T)  # Shape: (nu, N)

    # Convert data to PyTorch tensors
    # Stack the state trajectories (num_samples, nx, N+1) -> (num_samples * (N+1), nx)
    X_array = np.concatenate(X_init_val, axis=1).T  # Transpose to (num_samples * (N+1), nx)
    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    
    # Convert costs to tensor
    J_tensor = torch.tensor(J_X_init, dtype=torch.float32).view(-1, 1)
    
    # Initialize neural network with input size matching state dimension
    print(X_array.shape[1])
    net = NeuralNetwork(input_size=1, hidden_size=64, output_size=X_array.shape[1])
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    nn_func = net.create_casadi_function("NeuralNetwork", "./final_project/", X_array.shape[1], False)
    loss_fn = nn.MSELoss()
    J_terminal = []


    # Training loop
    for epoch in range(10):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        J_pred = net(torch.tensor(J_X_init[i], dtype=torch.float32))
        
        # Calculate loss
        loss = loss_fn(J_pred, torch.tensor(J_X_init[i], dtype=torch.float32))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:  # Print every 100 epochs
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    # After training, get the terminal cost prediction
    with torch.no_grad():
        X_init_tensor = torch.tensor(X_init, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        J_terminal = nn_func(X_init_tensor).item()  # Convert to Python scalar
        print(f"Final predicted terminal cost: {J_terminal:.4f}")
    '''M = 50
    J, X, U = create_and_solve_ocp(
        N, nx, nq, dt, x_init,
        10**log_w_v, 10**log_w_a, 10**log_w_w, 10**log_w_final, J_terminal, M)

    q_sol, dq_sol, ddq_sol, tau = extract_solution(J, X, U)
    
    # Print optimization results
    print("Optimization completed successfully!")
    print(f"Final cost: {J}")
    print(f"Initial state: {x_init}")
    print(f"Final state: {np.concatenate([q_sol[:, -1], dq_sol[:, -1]])}")
    
    # Display the motion
    print("Displaying robot motion...")
    display_motion(q_sol)

    # Plot results
    tt = np.linspace(0, (N + 1) * dt, N + 1)
    
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot joint positions
    for i in range(nq):
        ax1.plot(tt, q_sol[i, :].T, label=f'q {i+1}', alpha=0.7)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [rad]')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Joint Positions')

    # Plot joint velocities
    for i in range(nq):
        ax2.plot(tt, dq_sol[i, :].T, label=f'dq {i+1}', alpha=0.7)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [rad/s]')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Joint Velocities')

    # Plot joint torques (one time step shorter)
    tt_tau = tt[:-1]
    for i in range(nq):
        ax3.plot(tt_tau, tau[i, :], label=f'τ {i+1}', alpha=0.7)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Torque [Nm]')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Joint Torques')

    plt.tight_layout()
    plt.show()'''