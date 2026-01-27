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
device='cuda' if torch.cuda.is_available() else 'cpu'
# ====================== Robot and Dynamics Setup ======================
np.set_printoptions(precision=3, linewidth=200, suppress=True)
'''r = load('double_pendulum')
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
nq, nv = robot.nq, robot.nv'''
nq = 1
nx = 2 * nq

dt = 0.2
dt_N = 0.02
N = 500
# Initial state (random but within reasonable bounds)
 # Start from rest
q_des = np.zeros(nq)


# CasADi symbolic variables


m = 5.0
g = 9.81
I = 15.0

x = cs.SX.sym('x', nx)   # [q, dq]
u = cs.SX.sym('u', nq)   # [tau]

q  = x[0]
dq = x[1]
tau = u[0]

# dynamics
ddq = (tau + m*g*cs.sin(q)) / I

xdot = cs.vertcat(
    dq,
    ddq
)

# CasADi function
f = cs.Function('f', [x, u], [xdot])

q0 = np.zeros(nq)
dq0 = np.zeros(nq)
'''r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
simu = RobotSimulator(conf_ur5, r)
simu.init(q0, dq0)
simu.display(q0)'''
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
            cost += w_p * (X[k][:nq]).T @ (X[k][:nq]) * dt 
            cost += w_v * X[k][nq:].T @ X[k][nq:] * dt
            cost += w_a * U[k].T @ U[k] * dt

            # Add dynamics constraints
            opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))
            #opti.subject_to(X[-1][:nq] == q_des)
            # Add joint position limits (adjust these values as needed)
            #opti.subject_to(opti.bounded(-np.pi, X[k][:nq], np.pi))
            
            # Add joint velocity limits (adjust these values as needed)
            max_velocity = 5.0  # rad/s
            #opti.subject_to(opti.bounded(-max_velocity, X[k][nq:], max_velocity))
    else:
        for k in range(N):     
            # Compute cost function
            cost += w_p * (X[k][:nq]).T @ (X[k][:nq]) * dt 
            cost += w_v * X[k][nq:].T @ X[k][nq:] * dt
            cost += w_a * U[k].T @ U[k] * dt

            # Add dynamics constraints
            opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))
            #opti.subject_to(X[-1][:nq] == q_des)
            # Add joint position limits (adjust these values as needed)
            #opti.subject_to(opti.bounded(-np.pi, X[k][:nq], np.pi))
            
            # Add joint velocity limits (adjust these values as needed)
            max_velocity = 5.0  # rad/s
            #opti.subject_to(opti.bounded(-max_velocity, X[k][nq:], max_velocity))
            
    return cost


def define_terminal_cost_and_constraints(opti, X, q_des, w_p_final, J_terminal=None):
    terminal_cost = 0
    # Terminal cost from the neural network or quadratic cost
    if J_terminal is not None:
        terminal_cost = w_p_final * cs.sum1(J_terminal(X[-1]))
    # Add terminal constraint (optional but often useful)
    opti.subject_to(X[-1][:nq] == q_des)  # Uncomment to enforce terminal state exactly
    
    return terminal_cost


def create_and_solve_ocp(N, nx, nq, dt, x_init,
                         w_p, w_v, w_a, w_p_final, J_terminal=None, M=None):
    opti, X, U = create_decision_variables(N, nx, nq, M)
    running_cost = define_running_cost_and_dynamics(opti, X, U, N, dt, x_init, w_p, w_v, w_a, M)
    terminal_cost = define_terminal_cost_and_constraints(opti, X, q_des, w_p_final, J_terminal)
    '''if J_terminal is not None:
        terminal_cost += J_terminal'''
    print("terminal cost", terminal_cost)
    
    # Combine running and terminal costs
    #total_cost = running_cost + terminal_cost
    opti.minimize(running_cost + terminal_cost)

    opts = {
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.max_iter": 50000,
    "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 1e-3,
    "ipopt.acceptable_iter": 15,
    }
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    J_opt = sol.value(running_cost + terminal_cost)
    print(f"Solver time: {time.time() - t0:.2f}s")

 
    return sol, X, U, J_opt




def extract_solution(sol, X, U, M=None):
    if M is None:
        x_sol = np.array([np.array(sol.value(X[k])).flatten() for k in range(N + 1)])
        ddq_sol = np.array([np.array(sol.value(U[k])).flatten() for k in range(N)])
        q_sol = x_sol[:, :nq].T  # Shape: (nq, N+1)
        dq_sol = x_sol[:, nq:].T  # Shape: (nq, N+1)
        tau = np.zeros((nq, N))
        for i in range(N):
            for j in range(nq):
                # Handle 1D case for ddq_sol
                ddq_val = ddq_sol[i] if ddq_sol.ndim == 1 else ddq_sol[i, j]
                tau[j, i] = I * ddq_val + m * g * np.sin(q_sol[j, i])
    else:
        x_sol = np.array([np.array(sol.value(X[k])).flatten() for k in range(M + 1)])
        ddq_sol = np.array([np.array(sol.value(U[k])).flatten() for k in range(M)])
        q_sol = x_sol[:, :nq].T  # Shape: (nq, M+1)
        dq_sol = x_sol[:, nq:].T  # Shape: (nq, M+1)
        tau = np.zeros((nq, M))
        for i in range(M):
            for j in range(nq):
                # Handle 1D case for ddq_sol
                ddq_val = ddq_sol[i] if ddq_sol.ndim == 1 else ddq_sol[i, j]
                tau[j, i] = I * ddq_val + m * g * np.sin(q_sol[j, i])
    
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
    J_val = []
    for i in range(100):
        x_init = np.zeros(nx)
        x_init[:nq] = (np.random.rand(nq) - 0.5) * 0.05  # Small initial angles
        x_init[nq:] = (np.random.rand(nq) - 0.5) * 0.05 
        sol, X, U, J_opt = create_and_solve_ocp(
            N, nx, nq, dt, x_init,
            log_w_p, 10**log_w_v, 10**log_w_a, 10**log_w_final, None, None)
        q_sol, dq_sol, ddq_sol, tau = extract_solution(sol, X, U)
        print("Optimization completed successfully!")
        print(f"Final cost: {J_opt}")
        print(f"Initial state: {x_init}")
        print(f"Final state: {np.concatenate([q_sol[:, -1], dq_sol[:, -1]])}")
        # Extract the solution values
        x_sol = np.array([sol.value(X[k]) for k in range(N + 1)]).T  # Shape: (nx, N+1)
        J_val.append(J_opt)
        
        J_X_init.append(torch.tensor([J_opt], dtype=torch.float32))
        X_init_val.append(torch.tensor([x_init], dtype=torch.float32))
        U_init.append(np.array([sol.value(U[k]) for k in range(N)]).T)  # Shape: (nu, N)

    print("Average Optimized Cost on Running MPC", np.mean(J_val))

    net = NeuralNetwork(input_size=2, hidden_size=128, output_size=1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.000000001)
    print("input tensor shape hi", x_init)

    loss_fn = nn.MSELoss()
    J_terminal = []
    net.train()

    print("Any param requires grad:", next(net.parameters()).requires_grad)
    print("Model training mode:", net.training)


    t0 = time.time()
    # Training loop
    for i in range(100):
        # Zero the parameter gradients
        optimizer.zero_grad()
        print("input tensor shape", X_init_val[i])
        # Forward pass
        x = X_init_val[i].unsqueeze(0).to(device)  # (1, 4)
        J_pred = net(x)
        target = J_X_init[i].to(device)

        J_pred = J_pred.to(device)
        J_pred = J_pred.squeeze(0)
        print("J_pred shape", J_pred.shape)
        print("target shape", target.shape)
        print("J_pred.requires_grad:", J_pred.requires_grad)
        print("loss_fn input shapes:", J_pred.shape, target.shape)
       
        # Calculate loss
        loss = loss_fn(J_pred, target)
        # Backward pass and optimize

        loss.backward()
        optimizer.step()
        
        #if i % 100 == 0:  # Print every 100 epochs
        print(f'Epoch [{i+1}/1000], Loss: {loss.item():.4f}')
    print(f"Training time: {time.time() - t0:.2f}s")
    # After training, get the terminal cost prediction
    with torch.no_grad():
        #X_init_tensor = torch.tensor(X_init, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        nn_func = net.create_casadi_function("NeuralNetwork", "./final_project/", 2, True)   
        print(f"Final predicted terminal cost: {nn_func}")

    J_all_costs_without_terminal_with_M = []
    J_all_costs_with_terminal_with_M = []
    J_all_costs_without_terminal_with_N_M= []
    Costs_zero_with_M_without_terminal = []
    Costs_nonzero_with_M_without_terminal = []
    Costs_zero_with_M_with_terminal = []
    Costs_nonzero_with_M_with_terminal = []
    Costs_zero_with_N_M_without_terminal = []
    Costs_nonzero_with_N_M_without_terminal = []
    X_all = []
    
    for i in range(10):
        M = 5
        #Jterm_test = 1
        x_init_test = np.zeros(nx)
        x_init_test[:nq] = (np.random.rand(nq) - 0.5) * 0.05  # Small initial angles
        x_init_test[nq:] = (np.random.rand(nq) - 0.5) * 0.05
        X_all.append(x_init_test)
        log_w_final = 5
        sol, X, U, J = create_and_solve_ocp(
            N, nx, nq, dt, x_init_test,
            log_w_p, 10**log_w_v, 10**log_w_a, 10**log_w_final, None, M)
        q_sol, dq_sol, ddq_sol, tau = extract_solution(sol, X, U, M)
        # Print optimization results
        print(f"Optimization completed successfully for Test {i} with M = {M} and without Terminal Cost!")
        print(f"Final cost: {J}")
        final_state = np.concatenate([q_sol[:, -1], dq_sol[:, -1]])
        print(f"Initial state: {x_init_test}")
        print(f"Final state: {final_state}")
        norm = np.linalg.norm(final_state)

        if norm < 1e-3:
            label = "converged"
        elif norm < 5e-3:
            label = "near_equilibrium"
        else:
            label = "nonconverged"


        if label == "converged":
            # treat as non-zero final state
            Costs_zero_with_M_without_terminal.append(final_state)
        else:
            # treat as (effectively) zero
            Costs_nonzero_with_M_without_terminal.append(final_state)
        J_all_costs_without_terminal_with_M.append(J)
        
        # Display the motion
        #print("Displaying robot motion...")
        #display_motion(q_sol)

        # Plot results
        tt = np.linspace(0, (M + 1) * dt, M + 1)
        
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
        plt.show()



        sol, X, U, J = create_and_solve_ocp(
            N, nx, nq, dt, x_init_test,
            log_w_p, 10**log_w_v, 10**log_w_a, 10**log_w_final, nn_func, M)

        q_sol, dq_sol, ddq_sol, tau = extract_solution(sol, X, U, M)


        # Print optimization results
        print(f"Optimization completed successfully for Test {i} with M = {M} and Terminal Cost!")
        print(f"Final cost: {J}")
        final_state = np.concatenate([q_sol[:, -1], dq_sol[:, -1]])
        print(f"Initial state: {x_init_test}")
        print(f"Final state: {final_state}")
        norm = np.linalg.norm(final_state)

        if norm < 1e-3:
            label = "converged"
        elif norm < 5e-3:
            label = "near_equilibrium"
        else:
            label = "nonconverged"


        if label == "converged":
            # treat as non-zero final state
            Costs_zero_with_M_with_terminal.append(final_state)
        else:
            # treat as (effectively) zero
            Costs_nonzero_with_M_with_terminal.append(final_state)
        J_all_costs_with_terminal_with_M.append(J)
        # Display the motion
        #print("Displaying robot motion...")
        #display_motion(q_sol)

        # Plot results
        tt = np.linspace(0, (M + 1) * 10, M + 1)
        
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot joint positions
        for i in range(nq):
            ax1.plot(tt, q_sol[i, :], label=f'q {i+1}', alpha=0.7)
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
        plt.show()

        N = N + M

    
        sol, X, U, J = create_and_solve_ocp(
            N, nx, nq, dt, x_init_test,
            log_w_p, 10**log_w_v, 10**log_w_a, 10**log_w_final, None, None)

        q_sol, dq_sol, ddq_sol, tau = extract_solution(sol, X, U, None)

        # Print optimization results
        print(f"Optimization completed successfully for Test {i} with N+M  = {N} and without Terminal Cost!")
        print(f"Final cost: {J}")
        final_state = np.concatenate([q_sol[:, -1], dq_sol[:, -1]])
        print(f"Initial state: {x_init_test}")
        print(f"Final state: {final_state}")
        norm = np.linalg.norm(final_state)

        if norm < 1e-3:
            label = "converged"
        elif norm < 5e-3:
            label = "near_equilibrium"
        else:
            label = "nonconverged"


        if label == "converged":
            # treat as non-zero final state
            Costs_zero_with_N_M_without_terminal.append(final_state)
        else:
            # treat as (effectively) zero
            Costs_nonzero_with_N_M_without_terminal.append(final_state)
        J_all_costs_without_terminal_with_N_M.append(J)
        # Display the motion
        #print("Displaying robot motion...")
        #display_motion(q_sol)

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
        plt.show()
    J_avg_without_terminal_with_M = np.mean(J_all_costs_without_terminal_with_M)
    J_avg_with_terminal_with_M = np.mean(J_all_costs_with_terminal_with_M)
    J_avg_without_terminal_with_N_M = np.mean(J_all_costs_without_terminal_with_N_M)

    

    print("Average cost without terminal with M:", J_avg_without_terminal_with_M)
    print("Average cost with terminal with M:", J_avg_with_terminal_with_M)
    print("Average cost without terminal with N+M:", J_avg_without_terminal_with_N_M)

    total_zero_cost_without_terminal_with_M = len(Costs_zero_with_M_without_terminal)
    total_non_zero_cost_without_terminal_with_M = len(Costs_nonzero_with_M_without_terminal)

    total_zero_cost_with_terminal_with_M = len(Costs_zero_with_M_with_terminal)
    total_non_zero_cost_with_terminal_with_M = len(Costs_nonzero_with_M_with_terminal)

    total_zero_cost_without_terminal_with_N_M = len(Costs_zero_with_N_M_without_terminal)
    total_non_zero_cost_without_terminal_with_N_M = len(Costs_nonzero_with_N_M_without_terminal)

    print("Total count of without terminal with M and zero final state:", total_zero_cost_without_terminal_with_M)
    print("Total count of without terminal with M and nonzero final state:", total_non_zero_cost_without_terminal_with_M)
    
    print("Total count of with terminal with M and zero final state:", total_zero_cost_with_terminal_with_M)
    print("Total count of with terminal with M and nonzero final state:", total_non_zero_cost_with_terminal_with_M)
    
    print("Total count of without terminal with N+M and zero final state:", total_zero_cost_without_terminal_with_N_M)
    print("Total count of without terminal with N+M and nonzero final state:", total_non_zero_cost_without_terminal_with_N_M)
   
    
    