import numpy as np
import matplotlib.pyplot as plt

# Set font to Times New Roman before creating figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Total embedding dimension
d = 8  # We'll only *plot* the first 4 of these.

def pos_encoding(pos, dim, d=d):
    """
    Compute the positional encoding for:
      - 'pos' in [0..L-1] (the token position, 0-based)
      - 'dim' in [0..d-1] (the dimension index, 0-based)

    If dim = 2*k  => sin(pos * omega_k)
    If dim = 2*k+1 => cos(pos * omega_k)

    where omega_k = 1 / (10000^(2*k / d)).
    """
    k = dim // 2
    omega_k = 1.0 / (10000.0 ** ((2.0 * k) / d))
    if dim % 2 == 0:
        return np.sin(pos * omega_k)
    else:
        return np.cos(pos * omega_k)

# For plotting, we treat 'pos' as a continuous variable from 0 to 100
pos_min = 0
pos_max = 100
pos_plot = np.linspace(pos_min, pos_max, 300)

# We'll place 5 markers evenly in this range (to label them "The", "tallest", etc.)
marker_positions = np.linspace(pos_min, pos_max, 5)
x_labels = ["The", "tallest", "building", "in", "NYC"]

num_plots = 4  # We'll visualize only the first 4 dimensions j = 0,1,2,3

fig, axs = plt.subplots(num_plots, 1, figsize=(8, 14), sharex=True)

# We want the bottom subplot to show dim=0 and the top subplot to show dim=3
dims = list(range(num_plots))
dims_reversed = dims[::-1]

for ax, dim in zip(axs, dims_reversed):
    y_curve = [pos_encoding(p, dim, d) for p in pos_plot]
    y_markers = [pos_encoding(m, dim, d) for m in marker_positions]

    ax.plot(pos_plot, y_curve, color='black')
    ax.plot(marker_positions, y_markers, 'o', color='black')
    ax.set_yticks([-1, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove x-axis from all but the bottom plot
    if ax != axs[-1]:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xticks(marker_positions)
        ax.set_xticklabels(x_labels, fontsize=18)

    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('positional-encoding.png')
plt.show()

print("Y-coordinate values at marker positions:")
for pos_val, label in zip(marker_positions, x_labels):
    print(f"\nAt token '{label}' (pos = {pos_val:.2f}):")
    for dim in range(num_plots):
        val = pos_encoding(pos_val, dim, d)
        func_type = "sin" if dim % 2 == 0 else "cos"
        k = dim // 2
        omega = 1.0 / (10000.0 ** ((2.0 * k) / d))
        print(f"  Dimension {dim} ({func_type}, ω=1/(10000^(2*{k}/{d})) ≈ {omega:.5g}): {val:.5f}")
