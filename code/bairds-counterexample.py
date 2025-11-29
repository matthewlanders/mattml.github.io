import numpy as np
import matplotlib.pyplot as plt

alpha = 0.01
gamma = 0.99
theta = np.array([1, 1, 1, 1, 1, 10, 1])  # [theta_1, theta_2, ..., theta_7]


def V(s, theta):
    # Given the parameterization:
    # V(1) = 2*theta_1 + theta_7
    # V(2) = 2*theta_2 + theta_7
    # V(3) = 2*theta_3 + theta_7
    # V(4) = 2*theta_4 + theta_7
    # V(5) = 2*theta_5 + theta_7
    # V(6) = theta_6 + 5*theta_7
    if s in [1, 2, 3, 4, 5]:
        return 2 * theta[s - 1] + theta[6]
    elif s == 6:
        return theta[5] + 5 * theta[6]


def grad_V(s):
    g = np.zeros(7)
    if s in [1, 2, 3, 4, 5]:
        g[s - 1] = 2
        g[6] = 1
    elif s == 6:
        g[5] = 1
        g[6] = 5
    return g


cycle = [1, 2, 3, 4, 5, 6]
num_cycles = 20
theta_history = []

current_index = 0
current_state = cycle[current_index]

for t in range(num_cycles * len(cycle)):
    theta_history.append(theta.copy())

    next_s = 6
    R = 0

    delta = R + gamma * V(next_s, theta) - V(current_state, theta)
    g = grad_V(current_state)
    theta = theta + alpha * delta * g

    current_index = (current_index + 1) % len(cycle)
    current_state = cycle[current_index]

theta_history = np.array(theta_history)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.figure(figsize=(10, 8))
for i in range(7):
    plt.plot(theta_history[:, i], label=f"$\\theta_{i + 1}$", color='black')
plt.xlabel("Timesteps", fontsize=20)
plt.ylabel("Parameter Value", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('param-values-raw.png')
plt.show()
