import numpy as np

# TOMO:    radA = delta * [[1,1],[1,1],...]
# REGRESS: radA = delta * [[1,0],[1,0],...]

def interval_div(u_min, u_max, v_min, v_max):
    d = [u_min/v_min, u_min/v_max, u_max/v_min, u_max/v_max]
    return min(d), max(d)

def row_intervals(a, b, delta, mode):
    U = (a - delta, a + delta)
    V = (b - delta, b + delta) if mode == "tomo" else (b, b)
    return U, V

def lambda_interval(A0, delta, mode):
    a, b = A0[:,0], A0[:,1]
    lam_l, lam_r = -np.inf, np.inf
    for ai, bi in zip(a, b):
        U, V = row_intervals(ai, bi, delta, mode)
        if V[0] <= 0: return None
        lo, hi = interval_div(*U, *V)
        lam_l, lam_r = max(lam_l, lo), min(lam_r, hi)
        if lam_l > lam_r: return None
    return (lam_l, lam_r)

def construct_matrix(A0, delta, lam, mode, tol=1e-12):
    res = []
    for ai, bi in A0:
        U, V = row_intervals(ai, bi, delta, mode)
        WV = (min(lam*V[0], lam*V[1]), max(lam*V[0], lam*V[1]))
        lo, hi = max(U[0], WV[0]), min(U[1], WV[1])
        if lo > hi + tol: return None
        u = 0.5*(lo + hi)
        v = min(max(u/lam if abs(lam) > tol else 0, V[0]), V[1])
        res.append([u, v])
    return np.array(res)

def delta_star(A0, delta_init=0.05, delta_max=1.0, delta_tol=1e-4, mode="tomo"):
    # check delta=0
    lam_rng = lambda_interval(A0, 0, mode)
    if lam_rng: 
        lam = lam_rng[0]
        return 0, construct_matrix(A0, 0, lam, mode), lam

    # expand interval
    l, r = 0, delta_init
    while not lambda_interval(A0, r, mode) and r < delta_max:
        l, r = r, r*2
    if r >= delta_max: return None, None, None

    # bisection
    while r - l > delta_tol:
        m = 0.5*(l + r)
        if lambda_interval(A0, m, mode): r = m
        else: l = m

    lam_rng = lambda_interval(A0, r, mode)
    if not lam_rng: return None, None, None
    lam = lam_rng[0]
    A = construct_matrix(A0, r, lam, mode)
    if A is None and len(lam_rng) > 1:
        lam = lam_rng[1]; A = construct_matrix(A0, r, lam, mode)
    return r, A, lam

if __name__ == "__main__":
    A0 = np.array([[0.95,1.0],[1.05,1.0],[1.10,1.0]], float)

    for mode in ["tomo","regress"]:
        print(f"=== {mode.upper()} ===")
        delta_val, A, lam = delta_star(A0, delta_init=0.05, delta_tol=1e-5, mode=mode)
        if delta_val is None:
            print("Not found delta*.")
        else:
            print(f"delta* ≈ {delta_val:.6f}, lambda* ≈ {lam:.6f}")
            if A is not None:
                print("A*:") 
                for row in A:
                    print(" ".join(f"{x:8.5f}" for x in row))
