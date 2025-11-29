import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=1000)

# Histogram plot
plt.figure()
count, bins, ignored = plt.hist(data, 10, density=True, alpha=0.5, edgecolor='black', color='w')
plt.axvline(x=np.mean(data), color='gray', linestyle='dashed', linewidth=2)
plt.axvline(x=np.percentile(data, 99.7), color='black', linestyle='dashed', linewidth=2, label='99th Percentile')
plt.xticks([])
plt.yticks([])
plt.xlabel('Q(s,a)')
plt.ylabel('P(Q(s,a))')
plt.tight_layout()
plt.savefig('/Users/matt/src/mattml.github.io/static/images/iql/histogram.png')

# Parabola plot
plt.figure()
x = np.linspace(-10, 10, 400)
y_parabola = x**2
plt.axis('off')
plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
plt.plot(x, y_parabola, color='gray', linewidth=3.0)
plt.tight_layout()
plt.savefig('/Users/matt/src/mattml.github.io/static/images/iql/parabola.png')

# Expectile plot
plt.figure()
plt.axis('off')
tau = 0.9
y_expectile = np.where(x < 0, (1 - tau) * x**2, tau * x**2)
plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
plt.plot(x, y_expectile, color='black', linewidth=3.0)

plt.tight_layout()
plt.savefig('/Users/matt/src/mattml.github.io/static/images/iql/expectile.png')
