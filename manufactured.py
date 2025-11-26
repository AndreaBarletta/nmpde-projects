import sympy as sp

# --- Define symbols ---
x, y, t = sp.symbols('x y t')
H = sp.Function('H')(x, y, t)

# --- Define velocity field U(x,y,t) ---
# (Divergence-free, zero on boundary)
pi = sp.pi
u = 1
v = 0

# --- Define manufactured solution H(x,y,t) ---
H_man = sp.sin(2*pi*(x - t)) * sp.sin(2*pi*y)

# --- Compute derivatives ---
H_t = sp.diff(H_man, t)
H_x = sp.diff(H_man, x)
H_y = sp.diff(H_man, y)

# --- Compute source term S_H = dH/dt + U*grad(H) ---
S_H = +H_t + u*H_x + v*H_y

# --- Simplify the expression ---
S_H = sp.simplify(S_H)

# --- Print results ---
print("Velocity field u(x,y):", u)
print("Velocity field v(x,y):", v)
print("Manufactured solution H(x,y,t):", H_man)
print("Source term S_H(x,y,t):", S_H)
