import numpy as np
import intvalpy as ip
import matplotlib.pyplot as plt
import copy
import math
import os
from mpl_toolkits.mplot3d import Axes3D

ip.precision.extendedPrecisionQ = False

# --- Интервальные системы ---
A1 = ip.Interval([[[0.65, 1.25], [0.70, 1.30]],
                  [[0.75, 1.35], [0.70, 1.30]]])
b1 = ip.Interval([[0.96, 1.01], [1.00, 1.05]])

A2 = ip.Interval([[[0.65, 1.25], [0.70, 1.30]],
                  [[0.75, 1.35], [0.70, 1.30]],
                  [[0.80, 1.40], [0.70, 1.30]]])
b2 = ip.Interval([[0.96, 1.01], [1.00, 1.05], [1.02, 1.07]])

A3 = ip.Interval([[[0.65, 1.25], [0.70, 1.30]],
                  [[0.75, 1.35], [0.70, 1.30]],
                  [[0.80, 1.40], [0.70, 1.30]],
                  [[-0.30, 0.30], [0.70, 1.30]]])
b3 = ip.Interval([[0.96, 1.01], [1.00, 1.05], [1.02, 1.07], [0.63, 0.68]])

As = [A1, A2, A3]
bs = [b1, b2, b3]

# --- Проверка пустоты Tol ---
def is_empty(A, b):
    maxTol = ip.linear.Tol.maximize(A, b)
    return maxTol[1] < 0, maxTol[0], maxTol[1]

# --- b-коррекция ---
def b_correction(b, k):
    e = ip.Interval([[-k, k] for _ in range(len(b))])
    return b + e

def find_k_min(A, b, eps=1e-3, max_iter=50):
    emptiness, _, _ = is_empty(A, b)
    if not emptiness:
        return 0.0, b
    low, high = 0.0, None
    for i in range(max_iter):
        k_try = math.exp(i)
        b_try = b_correction(b, k_try)
        emptiness, _, _ = is_empty(A, b_try)
        if not emptiness:
            high = k_try
            low = math.exp(i - 1)
            break
    if high is None:
        raise RuntimeError("Не найдено k для разрешимости")
    iteration = 0
    while abs(high - low) > eps and iteration < max_iter:
        mid = (low + high) / 2
        b_mid = b_correction(b, mid)
        emptiness, _, _ = is_empty(A, b_mid)
        if emptiness:
            low = mid
        else:
            high = mid
        iteration += 1
    return high, b_correction(b, high)

# --- A-коррекция ---
def A_correction(A, b):
    max_tol = ip.linear.Tol.maximize(A, b)
    lower_bound = abs(max_tol[1]) / (sum(abs(max_tol[0])))
    rad_A = ip.rad(A)
    upper_bound = np.min(rad_A)
    e = (lower_bound + upper_bound) / 2
    corrected_A = []
    for i in range(len(A)):
        A_i = []
        for j in range(len(A[0])):
            if ip.rad(A[i][j]) == 0:
                A_i.append([A[i][j]._a, A[i][j]._b])
            else:
                A_i.append([A[i][j]._a + e, A[i][j]._b - e])
        corrected_A.append(A_i)
    return ip.Interval(corrected_A)

# --- Ab-коррекция ---
def Ab_correction(A, b, max_iter=50):
    new_A, new_b = copy.deepcopy(A), copy.deepcopy(b)
    emptiness, _, _ = is_empty(new_A, new_b)
    iteration = 0
    while emptiness and iteration < max_iter:
        iteration += 1
        new_A = A_correction(new_A, new_b)
        emptiness, _, _ = is_empty(new_A, new_b)
        if not emptiness:
            break
        new_b = b_correction(new_b, iteration)
        emptiness, _, _ = is_empty(new_A, new_b)
    return new_A, new_b

def visualize_tol(A, b, name: str):
    

    os.makedirs("img", exist_ok=True)
    max_tol = ip.linear.Tol.maximize(A, b)
    sol = max_tol[0]

    # --- 3D surface meshgrid 
    x3d = np.linspace(float(sol[0])-2, float(sol[0])+2, 70)
    y3d = np.linspace(float(sol[1])-2, float(sol[1])+2, 70)
    xx3d, yy3d = np.meshgrid(x3d, y3d)
    zz3d = np.array([[ip.linear.Tol.value(A, b, [xi, yi])
                      for xi, yi in zip(x_row, y_row)]
                     for x_row, y_row in zip(xx3d, yy3d)])

    # --- 2D functional meshgrid 
    x2d = np.linspace(0.0, 1.0, 100)
    y2d = np.linspace(0.0, 1.0, 100)
    xx2d, yy2d = np.meshgrid(x2d, y2d)
    zz2d = np.array([[1 if ip.linear.Tol.value(A, b, [xi, yi]) >= 0 else 0
                      for xi, yi in zip(x_row, y_row)]
                     for x_row, y_row in zip(xx2d, yy2d)])

    fig = plt.figure(figsize=(14,6))

    # 3D subplot 
    ax3d = fig.add_subplot(1,2,1, projection='3d')
    ax3d.plot_surface(xx3d, yy3d, zz3d, cmap='plasma', edgecolor='none', alpha=0.9)
    ax3d.scatter(sol[0], sol[1], float(max_tol[1]), color='red', s=60)
    ax3d.set_title(f"Tol surface ({name})")
    ax3d.set_box_aspect([1,1,0.6])  

    # 2D subplot 
    ax2d = fig.add_subplot(1,2,2)
    ax2d.contourf(xx2d, yy2d, zz2d, levels=1, colors=["#4472C4","#FFD966"], alpha=0.9)
    ax2d.scatter(sol[0], sol[1], color='black', marker='x', s=60)
    ax2d.set_title(f"Tol functional ({name})")
    ax2d.set_xlabel('$x_1$'); ax2d.set_ylabel('$x_2$')

    plt.tight_layout()
    plt.savefig(f"img/{name}.png")
    plt.close()

def run_all():
    corrections = [None, "A", "b", "Ab"]
    for i, (A_orig, b_orig) in enumerate(zip(As, bs)):
        print(f"\n=== System {i+1} ===")
        for corr in corrections:
            A = copy.deepcopy(A_orig)
            b = copy.deepcopy(b_orig)

            if corr == "b":
                k_min, b = find_k_min(A, b)
                print(f"[b-correction] k_min={k_min:.6g}")
            elif corr == "A":
                A = A_correction(A, b)
            elif corr == "Ab":
                A, b = Ab_correction(A, b)

            emptiness, argmax, maxTol = is_empty(A, b)
            name = f"S{i+1}_{corr}" if corr else f"S{i+1}_org"
            print(f"[{corr if corr else 'NoCorrection'}] Tol max={maxTol:.6f}, argmax={argmax}")
            visualize_tol(A, b, name)

# --- Запуск ---
run_all()