import numpy as np
import matplotlib.pyplot as plt
def f1(x):
  return x**3 - 3*x**2 + 2

def f2(x):
  return x**3 * np.sin(x) - x**2

x1 = np.linspace(0, 3, 400)
x2 = np.linspace(-2, 4, 400)
y1 = f1(x1)
y2 = f2(x2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(x1, y1, color='blue')
ax1.set_title('f1(x) = x^3 - 3x^2 + 2 на [0, 3]')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)

ax2.plot(x2, y2, color='red')
ax2.set_title('f2(x) = x^3 * sin(x) - x^2 на [-2, 4]')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

plt.tight_layout()
plt.show()