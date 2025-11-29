import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 400)
# Define multiple affine functions (y = mx + b)
def affine_func1(x): return 0.5 * x + 1
def affine_func2(x): return -0.3 * x + 10
def affine_func3(x): return 0.2 * x + 3
def affine_func4(x): return -0.7 * x + 15


y1 = affine_func1(x)
y2 = affine_func2(x)
y3 = affine_func3(x)
y4 = affine_func4(x)
y_infimum = np.minimum(np.minimum(y1, y2), np.minimum(y3, y4))

# Plot the affine functions
plt.plot(x, y1, '#0072B2')
plt.plot(x, y2, '#009E73')
plt.plot(x, y3, '#D55E00')
plt.plot(x, y4, '#CC79A7')

# Plot the bold line for the pointwise infimum
plt.plot(x, y_infimum, 'k', linewidth=2)
plt.tight_layout()
plt.savefig('/Users/matt/src/mattml.github.io/static/images/cmdps/affine', dpi=200, bbox_inches="tight")
