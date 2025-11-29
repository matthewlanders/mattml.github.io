import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 2*x**2 - 5*x + 6

def df(x):
    return 3*x**2 - 4*x - 5

# Generate x values
x = np.linspace(-2, 3, 400)
y = f(x)
x_points = [-0.5, 1.2, 2.05]

# Create a figure and a 3x1 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# Proportional
axs[0].plot(x, y, label="f(x) = $x^3 - 2x^2 - 5x + 6$", color='black')
for x_r in x_points:
    axs[0].plot([x_r, x_r], [0, f(x_r)], color='blue', linestyle="--", marker='o')
axs[0].axhline(0, color='black', lw=0.5)

# Integral
axs[1].plot(x, y, label="f(x) = $x^3 - 2x^2 - 5x + 6$", color='black')
axs[1].fill_between(x, y, color='gray', alpha=0.2)

# Derivative
tangent_length = 0.3
axs[2].plot(x, y, label="f(x) = $x^3 - 2x^2 - 5x + 6$", color='black')
for x_r in x_points:
    slope = df(x_r)
    c = f(x_r) - slope*x_r
    tangent_x = np.linspace(x_r - tangent_length, x_r + tangent_length, 100)
    tangent_line = slope*tangent_x + c
    axs[2].plot(tangent_x, tangent_line, linestyle="--", color='blue')

# Adjusting each subplot
for ax in axs:
    ax.axhline(0, color='black', lw=0.5)
    ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/matt/src/mattml.github.io/static/images/pid-lagrangian/plots.png')
plt.show()
