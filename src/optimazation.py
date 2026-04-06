import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

# def grad(f, x, dh):
#     px = (f([x[0] + dh, x[1]]) - f([x[0] - dh, x[1]]))/ (2.0 * dh)
#     pz = (f([x[0], x[1] + dh]) - f([x[0], x[1] - dh]))/ (2.0 * dh)
#     return np.array([px, pz], dtype=np.float32)

def grad(x):
    x1 = x[0]
    x2 = x[1]
    gx1 = -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1*x1)
    gx2 = 200.0 * (x2 - x1*x1)
    return np.array([gx1, gx2], dtype=np.float64)

def updateHessian(H, rho, s, y):
    I = np.eye(2, dtype=np.float32)
    A = I - rho * (s @ y.T)
    B = I - rho * (y @ s.T)
    H_new = A @ H @ B + rho * (s @ s.T)
    return H_new

def line_search(f, x, p, g, dh):
    alpha = 1.0
    c1 = 1e-4
    c2 = 0.9
    max = 30
    fx = f(x)
    for _ in range(max):
        x_new = x + alpha * p
        f_new = f(x_new)
        g_new = grad(x_new)

        armijo = f_new <= fx + c1 * alpha * g.T @ p
        curvature = abs(g_new.T @ p) <= c2 * abs(g.T @ p)

        if armijo and curvature:
            return alpha

        alpha *= 0.5

    return alpha
    
H = np.eye(2, dtype=np.float32)
ni = 50
dh = 1e-6
x0 = -1.5
z0 = -1.5
x = np.array([x0, z0], dtype=np.float32)
history = []
for _ in range(ni):
    g = grad(x)
    history.append((x[0], x[1]))
    p = -H @ g
    alpha = line_search(f,x,p,g,dh)
    x_new = x + alpha * p
    g_new = grad(x_new)
    s = (x_new - x).reshape(2, 1)
    y = (g_new - g).reshape(2, 1)
    ys = (y.T @ s)[0,0]
    if ys > 1e-12:
        rho = 1.0 / ys
        H = updateHessian(H, rho, s, y)
    g = g_new
    x = x_new

history.append((x[0], x[1]))
history = np.array(history)

x1 = np.linspace(-2.0, 2.0, 400)
x2 = np.linspace(-2.0, 2.0, 400)
X1, X2 = np.meshgrid(x1, x2)
F = f([X1, X2])

plt.figure()
plt.contourf(X1, X2, F, levels=30)
plt.plot(history[:,0], history[:,1], 'r.-', label='trajetória')
plt.plot(1, 1, 'bo', label='mínimo (1,1)')
plt.xlabel("x")
plt.ylabel("z")
plt.title("Rosenbrock")
plt.legend()
plt.grid()
plt.show()