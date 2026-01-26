import sympy as sp

def export_expression(expr):
    expr_c = sp.ccode(expr)
    expr_c = expr_c.replace('sin', 'std::sin')
    expr_c = expr_c.replace('cos', 'std::cos')
    expr_c = expr_c.replace('sqrt', 'std::sqrt')
    expr_c = expr_c.replace('pow', 'std::pow')
    expr_c = expr_c.replace('*', ' * ')
    return expr_c


def manufactured_H():
    x, y, t = sp.symbols('x y t')

    u = y * sp.sin(sp.pi * t) * sp.sin(sp.pi * x) * sp.cos(sp.pi/2 * y)
    v = sp.cos(sp.pi * t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)

    H = 1 + x * y * sp.cos(sp.pi * t)

    dtH = sp.diff(H, t)
    dxH = sp.diff(H, x)
    dyH = sp.diff(H, y)
    dxU = sp.diff(u, x)
    dyV = sp.diff(v, y)

    S_H = +dtH + u*dxH + v*dyH + H * (dxU + dyV)
    S_H = sp.simplify(S_H)

    print("Velocity field u(x,y):", export_expression(u))
    print("Velocity field v(x,y):", export_expression(v))
    print("Manufactured solution H(x,y,t):", export_expression(H))
    print("Source term S_H(x,y,t):", export_expression(S_H))

def manufactured_U():
    x, y, t, g, cf = sp.symbols('x y t g cf')

    u = y * sp.sin(sp.pi * t) * sp.sin(sp.pi * x) * sp.cos(sp.pi/2 * y)
    v = sp.cos(sp.pi * t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)

    H = 1 + x * y * sp.cos(sp.pi * t)

    u_x = sp.diff(u,x)
    u_y = sp.diff(u,y)
    u_t = sp.diff(u,t)

    v_x = sp.diff(v,x)
    v_y = sp.diff(v,y)
    v_t = sp.diff(v,t)

    H_x = sp.diff(H,x)
    H_y = sp.diff(H,y)

    U_norm = sp.sqrt(u**2 + v**2)

    S_u = u_t + u * u_x + v * u_y + g * H_x + cf/H * U_norm * u
    S_v = v_t + u * v_x + v * v_y + g * H_y + cf/H * U_norm * v

    # S_u = sp.simplify(S_u)
    # S_v = sp.simplify(S_v)

    print("Height H(x,y):", export_expression(H))
    print("Manufactured velocity field u(x,y,t):", export_expression(u))
    print("Manufactured velocity field v(x,y,t):", export_expression(v))
    print("Source term S_u(x,y,t):", export_expression(S_u))
    print("Source term S_v(x,y,t):", export_expression(S_v))

if __name__ == "__main__":
    print("Manufactured solution for H:")
    manufactured_H()
    print("\nManufactured solution for U:")
    manufactured_U()