import sympy as sp


def manufactured_H():
    x, y, t = sp.symbols('x y t')

    u = y * sp.sin(sp.pi * t) * sp.cos(sp.pi * x) * sp.cos(sp.pi/2 * y)
    v = sp.cos(sp.pi * t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * x)

    H_man = 1 + 0.5 * sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.cos(sp.pi * t)

    H_t = sp.diff(H_man, t)
    H_x = sp.diff(H_man, x)
    H_y = sp.diff(H_man, y)

    S_H = +H_t + u*H_x + v*H_y
    S_H = sp.simplify(S_H)

    print("Velocity field u(x,y):", u)
    print("Velocity field v(x,y):", v)
    print("Manufactured solution H(x,y,t):", H_man)
    print("Source term S_H(x,y,t):", S_H)

def manufactured_U():
    x, y, t, g, cf = sp.symbols('x y t g cf')

    u_man = y * sp.sin(sp.pi * t) * sp.cos(sp.pi * x) * sp.cos(sp.pi/2 * y)
    v_man = sp.cos(sp.pi * t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * x)

    H = 1 + 0.5 * sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.cos(sp.pi * t)

    u_x = sp.diff(u_man,x)
    u_y = sp.diff(u_man,y)
    u_t = sp.diff(u_man,t)

    v_x = sp.diff(v_man,x)
    v_y = sp.diff(v_man,y)
    v_t = sp.diff(v_man,t)

    H_x = sp.diff(H,x)
    H_y = sp.diff(H,y)

    U_norm = sp.sqrt(u_man**2 + v_man**2)

    S_u = u_t + u_man * u_x + v_man * u_y + g * H_x + cf/H * U_norm * u_man
    S_v = v_t + u_man * v_x + v_man * v_y + g * H_y + cf/H * U_norm * v_man

    S_u = sp.simplify(S_u)
    S_v = sp.simplify(S_v)

    print("Height H(x,y):", H)
    print("Manufactured velocity field u(x,y,t):", u_man)
    print("Manufactured velocity field v(x,y,t):", v_man)
    print("Source term S_u(x,y,t):", S_u)
    print("Source term S_v(x,y,t):", S_v)

if __name__ == "__main__":
    print("Manufactured solution for H:")
    manufactured_H()
    print("\nManufactured solution for U:")
    manufactured_U()