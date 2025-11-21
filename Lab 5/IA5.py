import numpy as np
import matplotlib.pyplot as plt
from intvalpy import Interval
from matplotlib.patches import Rectangle

# Interval vector class 2D
class IVec:
    """2D interval vector [x1] x [x2]"""
    def __init__(self, i1, i2):
        # Convert input to Interval if needed
        try:
            self.x1 = Interval(i1.a, i1.b)
        except AttributeError:
            if hasattr(i1, "__len__"):
                self.x1 = Interval(i1[0], i1[1])
            else:
                self.x1 = Interval(i1, i1)
        try:
            self.x2 = Interval(i2.a, i2.b)
        except AttributeError:
            if hasattr(i2, "__len__"):
                self.x2 = Interval(i2[0], i2[1])
            else:
                self.x2 = Interval(i2, i2)

    def mid(self):
        """Midpoint of the interval vector"""
        return np.array([(float(self.x1.a)+float(self.x1.b))/2,
                         (float(self.x2.a)+float(self.x2.b))/2])

    def width(self):
        """Width of intervals"""
        return np.array([float(self.x1.b - self.x1.a),
                         float(self.x2.b - self.x2.a)])

    def __repr__(self):
        return f"[{self.x1.a}, {self.x1.b}] × [{self.x2.a}, {self.x2.b}]"

# Functions f and interval Jacobian
def f_point(x, xc, yc=0.0):
    x1, x2 = x
    return np.array([
        x1 - x2**2,
        (x1 - xc)**2 + (x2 - yc)**2 - 1
    ])

def J_interval(X: IVec, xc, yc=0.0):
    """Interval Jacobian matrix"""
    a = Interval(1,1)
    b = Interval(-2,-2)*X.x2
    c = Interval(2,2)*(X.x1 - xc)
    d = Interval(2,2)*(X.x2 - yc)
    return np.array([[a,b],[c,d]], dtype=object)

# Krawczyk operator
def krawczyk(X: IVec, xc, yc=0.0):
    """Perform one Krawczyk iteration"""
    x0 = X.mid()
    fx0 = f_point(x0, xc, yc)
    J_int = J_interval(X, xc, yc)

    J_mid = np.array([
        [1, -2*x0[1]],
        [2*(x0[0]-xc), 2*(x0[1]-yc)]
    ], dtype=float)

    Y = np.linalg.inv(J_mid + 1e-12*np.eye(2))
    term1 = x0 - Y @ fx0

    IminusYJ = np.empty((2,2), dtype=object)
    for i in range(2):
        for j in range(2):
            acc = Interval(0,0)
            for k in range(2):
                acc += J_int[k,j]*Interval(Y[i,k], Y[i,k])
            base = Interval(1,1) if i==j else Interval(0,0)
            IminusYJ[i,j] = base - acc

    Xm = [X.x1 - x0[0], X.x2 - x0[1]]
    m0 = IminusYJ[0,0]*Xm[0] + IminusYJ[0,1]*Xm[1]
    m1 = IminusYJ[1,0]*Xm[0] + IminusYJ[1,1]*Xm[1]

    return IVec(Interval(term1[0], term1[0]) + m0,
                Interval(term1[1], term1[1]) + m1)

# Interval intersection
def intersect_IVec(A: IVec, B: IVec):
    lo1 = max(A.x1.a, B.x1.a)
    hi1 = min(A.x1.b, B.x1.b)
    lo2 = max(A.x2.a, B.x2.a)
    hi2 = min(A.x2.b, B.x2.b)
    if lo1 > hi1 or lo2 > hi2:
        return None
    return IVec([lo1, hi1],[lo2, hi2])

# Solve quartic for x2
def get_roots_x2(xc, yc=0.0):
    """Find real roots of (x2^2 - xc)^2 + (x2 - yc)^2 - 1 = 0"""
    coeff = [1, 0, 1 - 2*xc, 0, xc**2 - 1]
    roots = np.roots(coeff)
    x2_real = []
    for r in roots:
        if np.isreal(r):
            r = float(np.real(r))
            x2_real.append(r)
    return sorted(x2_real)

# Create initial interval X0 for each x2 root
def get_initial_intervals(xc, margin_x1=0.05, margin_x2=0.05):
    x2_roots = get_roots_x2(xc)
    intervals = []
    if not x2_roots:
        # No solution – take a wide interval
        intervals.append(IVec([-2,2], [-2,2]))
    else:
        for x2 in x2_roots:
            x1 = x2**2
            intervals.append(IVec([x1-margin_x1, x1+margin_x1],
                                   [x2-margin_x2, x2+margin_x2]))
    return intervals

# Run Krawczyk iterations and plot
xc_values = [0.0, 0.5, 1.0, 1.2]
yc = 0.0
count = 0
for xc in xc_values:
    X0_list = get_initial_intervals(xc)
    print(f"\n=== xc = {xc} ===")
    for idx, X0 in enumerate(X0_list):
        print(f"\n--- Initial interval #{idx+1} --- {X0}")
        print(f"Midpoint X0: x1={X0.mid()[0]:.6f}, x2={X0.mid()[1]:.6f}")
        iterations = [X0]
        Xcur = X0
        for k in range(3):
            K = krawczyk(Xcur, xc, yc)
            Xnext = intersect_IVec(K, Xcur)
            Xcur = Xnext if Xnext else K
            iterations.append(Xcur)

        # Print results for each iteration
        for i, box in enumerate(iterations):
            mid_x = box.mid()
            print(f"k={i}: X={box}, Midpoint=(x1={mid_x[0]:.6f}, x2={mid_x[1]:.6f})")

        # Plot contour f1=0 and f2=0
        margin = 0.5
        xs = np.linspace(float(X0.x1.a)-margin, float(X0.x1.b)+margin, 400)
        ys = np.linspace(float(X0.x2.a)-margin, float(X0.x2.b)+margin, 400)
        Xg, Yg = np.meshgrid(xs, ys)
        F1 = Xg - Yg**2
        F2 = (Xg - xc)**2 + (Yg - yc)**2 - 1

        fig, ax = plt.subplots(figsize=(6,5))
        ax.contour(Xg, Yg, F1, levels=[0], colors='blue', linewidths=1.5)
        ax.contour(Xg, Yg, F2, levels=[0], colors='red', linewidths=1.5)
        ax.set_title(f"Krawczyk Iterations x_c={xc}, interval #{idx+1})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        for k, box in enumerate(iterations):
            rect = Rectangle((box.x1.a, box.x2.a),
                             float(box.x1.b - box.x1.a),
                             float(box.x2.b - box.x2.a),
                             fill=False, edgecolor='green', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(box.x1.a, box.x2.b, f"X^{k}", fontsize=8, color='green')

        plt.tight_layout()        
        count+=1
        filename = f"5_{count}.png"
        plt.savefig(filename, dpi=300)
        print("Saved:", filename)
        plt.show()

