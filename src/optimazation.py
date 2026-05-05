import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x0 = x[0]
    x1 = x[1]
    y = 100.0 * (x1 - x0*x0)**2 + (x0 - 1)**2 
    return y

def grad(x):
    x0 = x[0]
    x1 = x[1]
    gx0 = 200.0*(x1 - x0*x0)*(-2.0*x0) + 2.0*(x0-1.0)
    gx1 = 200.0*(x1 - x0*x0)
    return np.array([gx0, gx1])

# def updateHessian(H, rho, s, y):
#     I = np.eye(2)
#     A = I - rho * (s @ y.T)
#     B = I - rho * (y @ s.T)
#     H_new = A @ H @ B + rho * (s @ s.T)
#     return H_new

def two_loop_recursion(g,s_store,y_store):
    q = g.copy()
    alpha = np.zeros(len(s_store))
    beta = np.zeros(len(s_store))

    for i in reversed(range(len(s_store))):
        s = s_store[i]
        y = y_store[i]
        sy = np.sum(s * y)
        rho = 1.0 / sy
        alpha[i] = rho * np.sum(s * q)
        q = q - alpha[i] * y

    if len(s_store) > 0:
        s_last = s_store[-1]
        y_last = y_store[-1]

        sy = np.sum(s_last * y_last)
        yy = np.sum(y_last * y_last)

        gamma = sy / yy

    else:
        gamma = 1.0

    r = gamma * q

    for i in range(len(s_store)):
        s = s_store[i]
        y = y_store[i]
        sy = np.sum(s * y)
        rho = 1.0 / sy
        beta[i] = rho * np.sum(y * r)
        r = r + s * (alpha[i] - beta[i])

    return r

# def line_search(x, p, g):
#     alpha = 1.0
#     c1 = 1e-4
#     c2 = 0.9
#     fx = f(x)
#     gTp = np.dot(g, p)

#     for _ in range(30):
#         x_new = x + alpha * p
#         f_new = f(x_new)
#         g_new = grad(x_new)
#         g_newTp = np.dot(g_new, p)
#         print("gTp",gTp)
#         print("X0",fx)    
#         print("X_new",f_new)
#         print("alpha",alpha)
#         print(fx + c1 * alpha * gTp)

#         armijo = f_new <= fx + c1 * alpha * gTp
#         curvature = abs(g_newTp) <= c2 * abs(gTp)

#         if armijo and curvature:
#             return alpha

#         alpha *= 0.5
#         print("alpha", alpha)

#     return alpha

# def line_search(x,p):
#     a0 = 0.0
#     a1 = 0.3
#     a2 = 1

#     x0 = x + a0 * p
#     x1 = x + a1 * p
#     x2 = x + a2 * p

#     f0 = f(x0)
#     f1 = f(x1)
#     f2 = f(x2)

#     num = (a1*a1 - a2*a2)*f0 + (a2*a2 - a0*a0)*f1 + (a0*a0 - a1*a1)*f2 
#     den = (a1 - a2)*f0 + (a2 - a0)*f1 + (a0 - a1)*f2

#     print("a1 =", a1, "a2 =", a2)
#     print("f0 =", f0)
#     print("f1 =", f1)
#     print("f2 =", f2)
#     print("num =", num)
#     print("den =", den)
#     alpha = 0.5*(num / den)
#     print("alpha",alpha)

#     return alpha

def line_search(x, p, g):
    c1 = 1e-4
    fx = f(x)
    gTp0 = np.dot(g, p)
    alpha = 1.0
    for it in range(30):
        x_new = x + alpha * p
        f_new = f(x_new)

        armijo = f_new <= fx + c1 * alpha * gTp0

        print("\n--- line_search_lo_hi ---")
        print("it =", it)
        print("alpha =", alpha)
        print("f0 =", fx)
        print("f_new =", f_new)
        print("gTp0 =", gTp0)
        print("Armijo =", armijo)

        if armijo:
            return alpha

        alpha *= 0.5 
    return alpha

ni = 30
x0 = -2.0
z0 = -1.5
x = np.array([x0, z0])
s_store = []
y_store = []
history = []
for _ in range(ni):
    g = grad(x)
    # g = g/np.max(np.abs(g))
    history.append((x[0], x[1]))
    p = -two_loop_recursion(g,s_store,y_store)
    alpha = line_search(x,p,g)
    x_new = x + alpha * p
    g_new = grad(x_new)
    # g_new = g_new/np.max(np.abs(g_new))
    s = (x_new - x)
    y = (g_new - g)
    s_store.append(s)
    y_store.append(y)
    x = x_new.copy()

history.append((x[0], x[1]))
history = np.array(history)

x1 = np.linspace(-2.0, 2.0, 400)
x2 = np.linspace(-2.0, 2.0, 400)
X1, X2 = np.meshgrid(x1, x2)
F = f([X1, X2])

plt.figure()
plt.contourf(X1, X2,F, levels=30)
plt.plot(history[:,0], history[:,1], 'r.-', label='trajetória')
plt.plot(1, 1, 'bo', label='mínimo (1,1)')
plt.xlabel("x")
plt.ylabel("z")
plt.title("Rosenbrock")
plt.legend()
plt.grid()
plt.show()