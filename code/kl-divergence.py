import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, stats, norm
from scipy.interpolate import interp1d

# Set the random seed for reproducibility
np.random.seed(0)

# Generate data for the first mode
mode1_mean = 8
mode1_std = 4
mode1_size = 1000
mode1_data = np.random.normal(mode1_mean, mode1_std, mode1_size)

# Generate data for the second mode
mode2_mean = 20
mode2_std = 3
mode2_size = 1000
mode2_data = np.random.normal(mode2_mean, mode2_std, mode2_size)

# Combine the two modes
data = np.concatenate((mode1_data, mode2_data))

# Estimate the kernel density function
kde = gaussian_kde(data)

# Generate points for the smooth curve
x_vals = np.linspace(min(data), max(data), 1000)
y_vals = kde(x_vals)

# Perform cubic interpolation to get a smooth line
interp_func = interp1d(x_vals, y_vals, kind='cubic')
smooth_x = np.linspace(min(data), max(data), 10000)
smooth_y = interp_func(smooth_x)
plt.axis('off')

plt.plot(smooth_x, smooth_y, lw=0.5, color='black')
plt.fill_between(smooth_x, smooth_y, color='gray', alpha=0.5)

# Forward
# x_axis = np.arange(-5, 30, 0.001)  # Plot between -5 and 30 with .001 steps.
# y_axis = norm.pdf(x_axis, 12, 6)  # Mean = 12, SD = 6
# plt.plot(x_axis, y_axis, lw=0.5, color='black')
# plt.fill_between(x_axis, y_axis, color='gray', alpha=0.2)
# plt.tight_layout()
# plt.savefig('forward-kl-divergence.png', dpi=300)
# plt.show()

# Reverse
x_axis = np.arange(1, 29, 0.001)
y_axis = norm.pdf(x_axis, 18, 3.5)

plt.plot(x_axis, y_axis, lw=0.5, color='black')
plt.fill_between(x_axis, y_axis, color='gray', alpha=0.2)
plt.tight_layout()
plt.savefig('reverse-kl-divergence.png', dpi=300)
plt.show()
