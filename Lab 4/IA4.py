import numpy as np
import struct
import intvalpy as ip
import matplotlib.pyplot as plt
import pandas as pd
import copy
ip.precision.extendedPrecisionQ = False

# --------------- CONFIG -----------------
FILE_X = "-0.205_lvl_side_a_fast_data.bin"
FILE_Y = "0.225_lvl_side_a_fast_data.bin"
ADC_DEN = 16384.0
RAD = 1.0 / (2 ** 14)
S_GRID_POINTS = 1000

# --------------- FILE READING -----------------
def read_bin_file(path):
    with open(path, 'rb') as f:
        header = f.read(256)
        frames = []
        point_dtype = np.dtype('<8H')
        side, mode, frame_count = struct.unpack('<BBH', header[:4])     
        for _ in range(frame_count):
            frame_header_data = f.read(16)
            if len(frame_header_data) < 16:
                break
            stop_point, timestamp = struct.unpack('<HL', frame_header_data[:6])
            frame_data = np.frombuffer(f.read(1024*16), dtype=point_dtype)
            frames.append(frame_data)
    frames = np.array(frames)
    volts = frames/ADC_DEN - 0.5
    return volts

# --------------- UTILITIES -----------------
def scalar_to_interval(x, rad):
    return ip.Interval(x-rad, x+rad)
scalar_to_interval_vec = np.vectorize(scalar_to_interval)

def get_avg(data):
    avg = np.zeros((1024,8))
    for i in range(len(data)): 
        avg = np.add(avg, data[i])
    return np.divide(avg, len(data))

# --- Jaccard ---
def jaccard_interval(x, y):
    a1, b1 = float(x.a), float(x.b)
    a2, b2 = float(y.a), float(y.b)
    numerator = min(b1, b2) - max(a1, a2)
    denominator = max(b1, b2) - min(a1, a2)
    return numerator / denominator

def coefficient_Jaccard(X, Y=None):
    if Y is None:
        # X là list
        lowers = [float(x.a) for x in X]
        uppers = [float(x.b) for x in X]
        return (min(uppers) - max(lowers)) / (max(uppers) - min(lowers))
    if isinstance(X, ip.ClassicalArithmetic) and isinstance(Y, ip.ClassicalArithmetic):
        return jaccard_interval(X, Y)
    return np.mean([jaccard_interval(x, y) for x, y in zip(X, Y)])

# --- Medians ---
def Mode(X, n_bins=200):
    lowers = np.array([float(el.a) for el in X])
    uppers = np.array([float(el.b) for el in X])
    grid = np.linspace(lowers.min(), uppers.max(), n_bins)
    counts = np.zeros(len(grid)-1)
    for i in range(len(grid)-1):
        counts[i] = np.sum((lowers <= grid[i+1]) & (uppers >= grid[i]))
    max_count = counts.max()
    bins = [(grid[i], grid[i+1]) for i in range(len(grid)-1) if counts[i]==max_count]
    merged = []
    curr = bins[0]
    for b in bins[1:]:
        if b[0] <= curr[1]:
            curr = (curr[0], b[1])
        else:
            merged.append(curr)
            curr = b
    merged.append(curr)
    return [ip.Interval(b[0], b[1]) for b in merged]

def MedK(x):
    lowers = [float(el.a) for el in x]
    uppers = [float(el.b) for el in x]
    med_lower = float(np.median(lowers))
    med_upper = float(np.median(uppers))
    return ip.Interval([med_lower, med_upper])

def MedP(x):
    X = sorted(x, key=lambda t: (float(t.a) + float(t.b)) / 2)
    index_med = len(X) // 2
    if len(X) % 2 == 0:
        return (X[index_med - 1] + X[index_med]) / 2
    return X[index_med]

# --- Plot optimization ---
def plot_opt(name, func, X, Y, lb, ub, model, n_points= S_GRID_POINTS):
    vals = np.linspace(lb, ub, n_points)
    Ji = [func(v, X, Y) for v in vals]
    idx = np.argmax(Ji)
    s_hat, Ji_max = vals[idx], Ji[idx]

    plt.figure(figsize=(8, 5))
    plt.plot(vals, Ji, 'b-', linewidth=2, label=f"{name}")
    plt.axvline(s_hat, color='red', linestyle='--', linewidth=2,
                label=f's_max = {s_hat:.6f}')
    plt.scatter([s_hat], [Ji_max], color='red', s=100, zorder=5)

    plt.xlabel('s (a)' if model == 'add' else 's (t)')
    plt.ylabel('F(s) = Jaccard')
    title_str = f"Method {name}" 
    plt.title(f"{title_str} - {model.upper()} Model\ns_hat = {s_hat:.6f}, Ji = {Ji_max:.6f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Ji_{name}_{model}.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return s_hat, Ji_max


def main():
    # ----- A -----
    x_data = get_avg(read_bin_file(FILE_X))
    y_data = get_avg(read_bin_file(FILE_Y))
    X = scalar_to_interval_vec(x_data, RAD).flatten()
    Y = scalar_to_interval_vec(y_data, RAD).flatten()
    print(f"Final dataset: {len(X)} samples")

    bound_a_l = float(np.min(Y).a) - float(np.max(X).b)
    bound_a_r = float(np.max(Y).b) - float(np.min(X).a)

    bound_t_l = float(np.min(Y).a) / float(np.max(X).b)
    bound_t_r = float(np.max(Y).b) / float(np.min(X).a)
    # ----- B -----
    models = ['add', 'mul']
    methods = ['B1', 'B2', 'B3', 'B4']
    method_names = {'B1': 'full', 'B2': 'mode', 'B3': 'medK', 'B4': 'medP'}
    # Define function dictionaries
    func_a_dict = {
        'B1': lambda a, X, Y: np.mean(coefficient_Jaccard(X + a, Y)),
        'B2': lambda a, X, Y: np.mean(coefficient_Jaccard(Mode(X + a), Mode(Y))),
        'B3': lambda a, X, Y: np.mean(coefficient_Jaccard(MedK(X + a), MedK(Y))),
        'B4': lambda a, X, Y: np.mean(coefficient_Jaccard(MedP(X + a), MedP(Y))),
    }

    func_t_dict = {
        'B1': lambda t, X, Y: np.mean(coefficient_Jaccard(X * t, Y)),
        'B2': lambda t, X, Y: np.mean(coefficient_Jaccard(Mode(X * t), Mode(Y))),
        'B3': lambda t, X, Y: np.mean(coefficient_Jaccard(MedK(X * t), MedK(Y))),
        'B4': lambda t, X, Y: np.mean(coefficient_Jaccard(MedP(X * t), MedP(Y))),
    }

    all_results = {}

    for model in models:
        print(f"\n=== Optimizing for {model.upper()} model ===")
        model_results = {}

        for method in methods:
            print(f"--- Method {method} ({method_names[method]}) ---")
            if model == 'add':
                func = func_a_dict[method]
                lb = bound_a_l
                ub = bound_a_r
            else:
                func = func_t_dict[method]
                lb = -1.2
                ub = bound_t_r
            s_hat, Ji_max = plot_opt(method, func, X, Y, lb, ub, model)

            model_results[method] = {'s_hat': s_hat, 'Ji_max': Ji_max}
            print(f"  s_hat = {s_hat:.6f}, Ji_max = {Ji_max:.6f}")


        all_results[model] = model_results

    # --- D ---
    print("\n=== Summary Table ===")
    print("Method | Add Model (a) | Ji_add | Mul Model (t) | Ji_mul")
    print("-------|---------------|--------|---------------|--------")
    for method in methods:
            add_result = all_results['add'][method]
            mul_result = all_results['mul'][method]
            print(f"{method:6} | {add_result['s_hat']:13.6f} | {add_result['Ji_max']:6.4f} | "
                  f"{mul_result['s_hat']:13.6f} | {mul_result['Ji_max']:6.4f}")
    
    best_add = max(methods, key=lambda m: all_results['add'][m]['Ji_max'])
    best_mul = max(methods, key=lambda m: all_results['mul'][m]['Ji_max'])

    print(f"Best method for ADD model: {best_add} (Ji = {all_results['add'][best_add]['Ji_max']:.6f})")
    print(f"Best method for MUL model: {best_mul} (Ji = {all_results['mul'][best_mul]['Ji_max']:.6f})")

    add_values = [all_results['add'][m]['s_hat'] for m in methods]
    mul_values = [all_results['mul'][m]['s_hat'] for m in methods]

    print(f"Variation in ADD model estimates: {np.std(add_values):.6f}")
    print(f"Variation in MUL model estimates: {np.std(mul_values):.6f}")

if __name__ == "__main__":
    main()
