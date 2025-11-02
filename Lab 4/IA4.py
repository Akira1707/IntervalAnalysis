import struct
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

# --------------- CONFIG -----------------
FILE_X = "-0.205_lvl_side_a_fast_data.bin"
FILE_Y = "0.225_lvl_side_a_fast_data.bin"
ADC_DEN = 16384      
RAD = 1.0 / (2 ** 14)   
EPS = 1e-4              
S_GRID_POINTS = 2000    
CHANNEL = None          

# Binary format constants
FILE_HEADER_SIZE = 256
FRAME_SIZE = 16400
FRAME_HEADER_SIZE = 16
POINTS_PER_FRAME = 1024
CHANNELS = 8  

def read_bin_file(path: str):
    with open(path, "rb") as f:
        file_header = f.read(FILE_HEADER_SIZE)
        if len(file_header) < FILE_HEADER_SIZE:
            raise ValueError("File too small or header missing.")
        side, mode, frame_count = struct.unpack_from("<BBH", file_header, 0)
        frames = []
        for fr in range(frame_count):
            frame_block = f.read(FRAME_SIZE)
            if len(frame_block) < FRAME_SIZE:
                print(f"Warning: expected frame block size {FRAME_SIZE}, got {len(frame_block)} at frame {fr}")
                break
            payload = frame_block[FRAME_HEADER_SIZE:]
            arr = np.frombuffer(payload, dtype=np.uint16)
            if arr.size != POINTS_PER_FRAME * CHANNELS:
                raise ValueError(f"Unexpected payload size (words): {arr.size}")
            arr = arr.reshape((POINTS_PER_FRAME, CHANNELS))
            frames.append(arr)
        if len(frames) == 0:
            return np.empty((0, POINTS_PER_FRAME, CHANNELS), dtype=np.uint16)
        return np.stack(frames, axis=0)  

def adc_to_volts(arr_codes: np.ndarray, denom=ADC_DEN) -> np.ndarray:
    return arr_codes.astype(np.float64) / float(denom) - 0.5

def make_intervals_from_volts(volts: np.ndarray, rad=RAD):
    lows = volts - rad
    highs = volts + rad
    return lows, highs

def flatten_frames_to_samples(frames: np.ndarray, channel=None):
    if channel is None:
        flat = frames.reshape(-1) 
    else:
        flat = frames[:, :, channel].reshape(-1)
    return flat

def interval_intersection_length(a_lo, a_hi, b_lo, b_hi):
    lo = np.maximum(a_lo, b_lo)
    hi = np.minimum(a_hi, b_hi)
    inter = np.maximum(0.0, hi - lo)
    return inter

def interval_union_length(a_lo, a_hi, b_lo, b_hi):
    lo = np.minimum(a_lo, b_lo)
    hi = np.maximum(a_hi, b_hi)
    union = np.maximum(0.0, hi - lo)
    return union

def transform_add(a, x_lo, x_hi):
    return x_lo + a, x_hi + a

def transform_mul(t, x_lo, x_hi):
    lo = t * x_lo
    hi = t * x_hi
    new_lo = np.minimum(lo, hi)
    new_hi = np.maximum(lo, hi)
    return new_lo, new_hi

def jaccard_for_s(s, X_lo, X_hi, Y_lo, Y_hi, model='add'):
    if model == 'add':
        TX_lo, TX_hi = transform_add(s, X_lo, X_hi)
    elif model == 'mul':
        TX_lo, TX_hi = transform_mul(s, X_lo, X_hi)
    else:
        raise ValueError("model must be 'add' or 'mul'")

    inter = interval_intersection_length(TX_lo, TX_hi, Y_lo, Y_hi)
    union = interval_union_length(TX_lo, TX_hi, Y_lo, Y_hi)
    valid_union = union > 0
    if not np.any(valid_union):
        return 0.0
    numer = np.sum(inter[valid_union])
    denom = np.sum(union[valid_union])
    return numer / denom if denom > 0 else 0.0

def interval_mode(lows, highs, nbins=1000):
    if len(lows) == 0:
        return 0.0, 0.0
    lo_min = float(np.min(lows))
    hi_max = float(np.max(highs))
    if hi_max - lo_min < 1e-12:
        return lo_min, hi_max
    grid = np.linspace(lo_min, hi_max, nbins)
    coverage = np.zeros_like(grid, dtype=int)
    for l, h in zip(lows, highs):
        coverage += ((grid >= l) & (grid <= h)).astype(int)
    maxcov = coverage.max()
    if maxcov == 0:
        return lo_min, lo_min
    mask = (coverage == maxcov)
    runs = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return float(grid[0]), float(grid[0])
    best = max(runs, key=lambda r: r[1]-r[0])
    return float(grid[best[0]]), float(grid[best[1]])

def interval_median_medK(lows, highs):
    lo = float(np.median(lows))
    hi = float(np.median(highs))
    if lo > hi:
        m = (lo + hi) / 2
        return m, m
    return lo, hi

def interval_median_medP(lows, highs):
    mids = (lows + highs) / 2.0
    median_mid = float(np.median(mids))
    avg_rad = float(np.mean((highs - lows) / 2.0))
    return median_mid - avg_rad, median_mid + avg_rad

def Ji_single_interval(s, Xlo, Xhi, Ylo, Yhi, model='add'):
    return jaccard_for_s(s, np.array([Xlo]), np.array([Xhi]), np.array([Ylo]), np.array([Yhi]), model=model)

def optimize_parameter(X_lo, X_hi, Y_lo, Y_hi, model='add', method='full', s_range=None, grid_points=1000):
    if method == 'full':
        X_lo_used, X_hi_used = X_lo, X_hi
        Y_lo_used, Y_hi_used = Y_lo, Y_hi
        jaccard_func = jaccard_for_s
    else:
        if method == 'mode':
            stat_x = interval_mode(X_lo, X_hi)
            stat_y = interval_mode(Y_lo, Y_hi)
        elif method == 'medK':
            stat_x = interval_median_medK(X_lo, X_hi)
            stat_y = interval_median_medK(Y_lo, Y_hi)
        elif method == 'medP':
            stat_x = interval_median_medP(X_lo, X_hi)
            stat_y = interval_median_medP(Y_lo, Y_hi)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_lo_used, X_hi_used = np.array([stat_x[0]]), np.array([stat_x[1]])
        Y_lo_used, Y_hi_used = np.array([stat_y[0]]), np.array([stat_y[1]])
        jaccard_func = jaccard_for_s

    if s_range is None:
        if model == 'add':
            smin = float(np.min(Y_lo_used) - np.max(X_hi_used))
            smax = float(np.max(Y_hi_used) - np.min(X_lo_used))
        else:
            safe_x_min = np.percentile((X_lo_used + X_hi_used) / 2.0, 1)
            safe_x_max = np.percentile((X_lo_used + X_hi_used) / 2.0, 99)
            if abs(safe_x_min) < 1e-9:
                safe_x_min += 1e-6
            smin = float(np.min(Y_lo_used) / safe_x_max)
            smax = float(np.max(Y_hi_used) / max(safe_x_min, 1e-6))
        
        if smin == smax:
            smin -= 1.0
            smax += 1.0
    else:
        smin, smax = s_range

    grid = np.linspace(smin, smax, grid_points)
    vals = np.zeros_like(grid)
    
    for i, s in enumerate(grid):
        vals[i] = jaccard_func(s, X_lo_used, X_hi_used, Y_lo_used, Y_hi_used, model=model)

    idx_max = np.argmax(vals)
    s_hat = grid[idx_max]
    Ji_max = vals[idx_max]
    
    return s_hat, Ji_max, grid, vals

def plot_results_all_methods(results: Dict[str, Dict], model: str, filename: str):
    plt.figure(figsize=(12, 8))
    
    colors = {'B1': 'blue', 'B2': 'red', 'B3': 'green', 'B4': 'orange'}
    labels = {'B1': 'Full data', 'B2': 'Mode', 'B3': 'MedK', 'B4': 'MedP'}
    
    for method, result in results.items():
        if method in colors:
            plt.plot(result['grid'], result['vals'], 
                    color=colors[method], label=labels[method], linewidth=2)
            plt.axvline(result['s_hat'], color=colors[method], 
                       linestyle='--', alpha=0.7)
            plt.scatter([result['s_hat']], [result['Ji_max']], 
                       color=colors[method], s=100, zorder=5)

    plt.xlabel('s (a)' if model == 'add' else 's (t)', fontsize=12)
    plt.ylabel('F(s) = Jaccard', fontsize=12)
    plt.title(f'Comparison of All Methods - {model.upper()} Model', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # ----- A -----
    print("=== A. Reading and preparing data ===")
    frames_x = read_bin_file(FILE_X)
    frames_y = read_bin_file(FILE_Y)
    
    codes_x = flatten_frames_to_samples(frames_x, channel=CHANNEL)
    codes_y = flatten_frames_to_samples(frames_y, channel=CHANNEL)
    
    volts_x = adc_to_volts(codes_x)
    volts_y = adc_to_volts(codes_y)
    
    X_lo, X_hi = make_intervals_from_volts(volts_x, rad=RAD)
    Y_lo, Y_hi = make_intervals_from_volts(volts_y, rad=RAD)
    
    m = min(len(X_lo), len(Y_lo))
    if len(X_lo) != len(Y_lo):
        print(f"Warning: Different sample sizes - X: {len(X_lo)}, Y: {len(Y_lo)}, using {m} samples")
        X_lo, X_hi = X_lo[:m], X_hi[:m]
        Y_lo, Y_hi = Y_lo[:m], Y_hi[:m]
    
    print(f"Final dataset: {len(X_lo)} samples")
    
    # ----- B -----
    models = ['add', 'mul']
    methods = ['B1', 'B2', 'B3', 'B4']
    method_names = {'B1': 'full', 'B2': 'mode', 'B3': 'medK', 'B4': 'medP'}
    
    all_results = {}
    
    for model in models:
        print(f"\n=== Optimizing for {model.upper()} model ===")
        model_results = {}
        
        for method in methods:
            print(f"--- Method {method} ({method_names[method]}) ---")
            
            s_hat, Ji_max, grid, vals = optimize_parameter(
                X_lo, X_hi, Y_lo, Y_hi, 
                model=model, 
                method=method_names[method],
                grid_points=1000
            )
            
            model_results[method] = {
                's_hat': s_hat,
                'Ji_max': Ji_max,
                'grid': grid,
                'vals': vals
            }
            
            print(f"  s_hat = {s_hat:.6f}, Ji_max = {Ji_max:.6f}")
            
            plt.figure(figsize=(8, 5))
            plt.plot(grid, vals, 'b-', linewidth=2)
            plt.axvline(s_hat, color='red', linestyle='--', linewidth=2, 
                       label=f's_max = {s_hat:.6f}')
            plt.scatter([s_hat], [Ji_max], color='red', s=100, zorder=5)
            plt.xlabel('s (a)' if model == 'add' else 's (t)')
            plt.ylabel('F(s) = Jaccard')
            plt.title(f'Method {method} - {model.upper()} Model\ns_hat = {s_hat:.6f}, Ji = {Ji_max:.6f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"Ji_{method}_{model}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        all_results[model] = model_results
        
        plot_results_all_methods(model_results, model, f"comparison_{model}.png")
    
    print("\n=== D. Comparison of Results ===")
    
    print("\n--- ADD Model Results ---")
    for method in methods:
        result = all_results['add'][method]
        print(f"{method}: s_hat = {result['s_hat']:.6f}, Ji = {result['Ji_max']:.6f}")
    
    print("\n--- MUL Model Results ---")
    for method in methods:
        result = all_results['mul'][method]
        print(f"{method}: s_hat = {result['s_hat']:.6f}, Ji = {result['Ji_max']:.6f}")
    
    print("\n=== Summary Table ===")
    print("Method | Add Model (a) | Ji_add | Mul Model (t) | Ji_mul")
    print("-------|---------------|--------|---------------|--------")
    for method in methods:
        add_result = all_results['add'][method]
        mul_result = all_results['mul'][method]
        print(f"{method:6} | {add_result['s_hat']:13.6f} | {add_result['Ji_max']:6.4f} | "
              f"{mul_result['s_hat']:13.6f} | {mul_result['Ji_max']:6.4f}")
    
    print("\n=== Analysis ===")
    
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