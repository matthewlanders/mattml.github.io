import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

x = np.linspace(0, 1, 1000)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.figure(figsize=(8, 5))
plt.plot(x, beta.pdf(x, 2, 2), color='gray')
plt.plot(x, beta.pdf(x, 30, 30), color='black')

plt.xlabel(r'$\theta$', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig('encoding-confidence.png', bbox_inches="tight")
plt.show()
