import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

x = np.linspace(0, 1, 1000)
params = [
    (1, 1, 'Uniform Prior', 'black'),
    (2, 2, 'Weak belief: Centered at 0.5', 'dimgray'),
    (10, 10, 'Strong belief: Concentrated at 0.5', 'gray'),
    (5, 2, 'Skewed belief: $\mathbb{E}[\Theta] > 0.5$', 'slategray'),
    (2, 5, 'Skewed belief: $\mathbb{E}[\Theta] < 0.5$', 'darkgray')
]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.figure(figsize=(8, 5))
for a, b, label, color in params:
    plt.plot(x, beta.pdf(x, a, b), label=label, color=color)

plt.xlabel(r'$\theta$', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig('beta-priors.png', bbox_inches="tight")
plt.show()
