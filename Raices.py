import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Se añade para la creación de tablas


# Configuración de gráficos
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 10
TOL = 1e-6
MAX_ITER = 100

def imprimir_corrida(iteraciones, titulo, metodo):
    """Imprime el historial de iteraciones en formato de tabla (corrida)."""
    if not iteraciones:
        print(f"\n--- Historial de Iteraciones ({metodo}) ---")
        print("No se generó historial de iteraciones.")
        return

    if metodo == "Bisección":
        df = pd.DataFrame(iteraciones, columns=['i', 'a', 'b', 'c', 'f(a)', 'f(b)', 'f(c)', 'Error_a'])
    elif metodo == "Newton-Raphson":
        df = pd.DataFrame(iteraciones, columns=['i', 'xi', 'f(xi)', "f'(xi)", 'x_new', 'Error_a'])
    elif metodo == "Secante":
        df = pd.DataFrame(iteraciones, columns=['i', 'x_prev', 'x_curr', 'f_prev', 'f_curr', 'x_next', 'Error_a'])
    
    print(f"\n--- CORRIDA: {metodo} ({titulo}) ---")
    print(df.round(6))

# =============================================
# IMPLEMENTACIÓN DE MÉTODOS
# =============================================

def metodo_biseccion(f, a, b, tol=TOL, max_iter=MAX_ITER):
    """Método de bisección con historial de iteraciones completo."""
    if f(a) * f(b) >= 0:
        # Se devuelve un mensaje especial en lugar de None para reportar el error.
        return None, "Error: No hay cambio de signo en el intervalo"
    
    iteraciones = []
    for i in range(max_iter):
        c = (a + b) / 2
        error_a = abs(b - a) / 2 # Error en bisección es la mitad del intervalo
        iteraciones.append((i+1, a, b, c, f(a), f(b), f(c), error_a))
        
        if abs(f(c)) < tol or error_a < tol:
            return c, iteraciones
            
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
            
    return c, iteraciones

def metodo_newton(f, df, x0, tol=TOL, max_iter=MAX_ITER):
    """Método de Newton-Raphson con historial de iteraciones completo."""
    iteraciones = []
    x = x0
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-12:
            return None, "Error: Derivada cercana a cero"
            
        x_new = x - fx / dfx
        error_a = abs(x_new - x)
        iteraciones.append((i+1, x, fx, dfx, x_new, error_a))
        
        if error_a < tol or abs(fx) < tol:
            return x_new, iteraciones
            
        x = x_new
            
    return x, iteraciones

def metodo_secante(f, x0, x1, tol=TOL, max_iter=MAX_ITER):
    """Método de la secante con historial de iteraciones completo."""
    iteraciones = []
    x_prev, x_curr = x0, x1
    f_prev, f_curr = f(x0), f(x1)
    
    for i in range(max_iter):
        if abs(f_curr - f_prev) < 1e-12:
            return None, "Error: Diferencia de funciones cercana a cero"
            
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        error_a = abs(x_next - x_curr)
        iteraciones.append((i+1, x_prev, x_curr, f_prev, f_curr, x_next, error_a))
        
        if error_a < tol or abs(f_curr) < tol:
            return x_next, iteraciones
            
        x_prev, x_curr = x_curr, x_next
        f_prev, f_curr = f_curr, f(x_next)
            
    return x_curr, iteraciones

def graficar_funcion(f, raices, intervalo, titulo, num_ejercicio, save_name=None):
    """Graficar función y raíces encontradas."""
    x = np.linspace(intervalo[0], intervalo[1], 1000)
    y = f(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label=f'f(x)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Marcar raíces
    for i, raiz in enumerate(raices):
        if isinstance(raiz, (float, np.float64)) and intervalo[0] <= raiz <= intervalo[1]:
            plt.plot(raiz, f(raiz), 'ro', markersize=8, label=f'Raíz {i+1}: x ≈ {raiz:.6f}')
            plt.text(raiz, f(raiz), f' ({raiz:.3f}, 0)', ha='right', va='bottom')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Ejercicio {num_ejercicio}: {titulo}')
    plt.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    plt.show()

# =============================================
# EJERCICIO 1: x³ - e^(0.8x) = 20
# =============================================
print("=" * 60)
print("EJERCICIO 1: x³ - e^(0.8x) = 20")
print("Se busca la única raíz real en [0, 8].")
print("=" * 60)

def f1(x):
    return x**3 - np.exp(0.8*x) - 20

def df1(x):
    return 3*x**2 - 0.8*np.exp(0.8*x)

# Raíz única esperada cerca de 3.01
# Aplicar métodos al intervalo [2, 4]
a1, b1 = 2, 4

raiz1_bisec, iter_bisec1 = metodo_biseccion(f1, a1, b1)
raiz1_newton, iter_newton1 = metodo_newton(f1, df1, 3.5)
raiz1_secante, iter_secante1 = metodo_secante(f1, 2.5, 3.5)

print(f"\n--- SOLUCIONES (Raíz 1) ---")
print(f"Bisección: {raiz1_bisec:.6f}")
print(f"Newton: {raiz1_newton:.6f}")
print(f"Secante: {raiz1_secante:.6f}")

# Corridas (Tablas)
imprimir_corrida(iter_bisec1, "Raíz 1", "Bisección")
imprimir_corrida(iter_newton1, "Raíz 1", "Newton-Raphson")
imprimir_corrida(iter_secante1, "Raíz 1", "Secante")

# Graficar
graficar_funcion(f1, [raiz1_bisec], [0, 8], "x³ - e^(0.8x) = 20 (Raíz única)", 1)

# =============================================
# EJERCICIO 2: 3sin(0.5x) - 0.5x + 2 = 0
# =============================================
print("\n" + "=" * 60)
print("EJERCICIO 2: 3sin(0.5x) - 0.5x + 2 = 0")
print("Intervalo de búsqueda: [5, 6]")
print("=" * 60)

def f2(x):
    return 3*np.sin(0.5*x) - 0.5*x + 2

def df2(x):
    return 1.5*np.cos(0.5*x) - 0.5

# Aplicar métodos al intervalo [5, 6]
a2, b2 = 5, 6

raiz_bisec2, iter_bisec2 = metodo_biseccion(f2, a2, b2)
raiz_newton2, iter_newton2 = metodo_newton(f2, df2, 5.5)
raiz_secante2, iter_secante2 = metodo_secante(f2, a2, b2)

print(f"\n--- SOLUCIONES ---")
print(f"Bisección: {raiz_bisec2:.6f}")
print(f"Newton: {raiz_newton2:.6f}")
print(f"Secante: {raiz_secante2:.6f}")

# Corridas (Tablas)
imprimir_corrida(iter_bisec2, "Raíz", "Bisección")
imprimir_corrida(iter_newton2, "Raíz", "Newton-Raphson")
imprimir_corrida(iter_secante2, "Raíz", "Secante")

# Graficar
graficar_funcion(f2, [raiz_bisec2], [0, 8], "3sin(0.5x) - 0.5x + 2 = 0", 2)

# =============================================
# EJERCICIO 3: x³ - x²e^(-0.5x) - 3x = -1
# =============================================
print("\n" + "=" * 60)
print("EJERCICIO 3: x³ - x²e^(-0.5x) - 3x = -1")
print("Tres raíces reales.")
print("=" * 60)

def f3(x):
    return x**3 - (x**2)*np.exp(-0.5*x) - 3*x + 1

def df3(x):
    # Derivada simplificada: 3x² - e^(-0.5x) * (2x - 0.5x²) - 3
    return 3*x**2 - np.exp(-0.5*x) * (2*x - 0.5*x**2) - 3

# Raíz 1: [-2, -1]
raiz1_bisec3, _ = metodo_biseccion(f3, -2, -1)
raiz1_newton3, _ = metodo_newton(f3, df3, -1.5)
raiz1_secante3, iter_secante3 = metodo_secante(f3, -2, -1)

# Raíz 2: [0, 1]
raiz2_bisec3, _ = metodo_biseccion(f3, 0, 1)
raiz2_newton3, _ = metodo_newton(f3, df3, 0.5)
raiz2_secante3, iter_secante3_2 = metodo_secante(f3, 0, 1)

# Raíz 3: [1, 2]
raiz3_bisec3, iter_bisec3_3 = metodo_biseccion(f3, 1, 2)
raiz3_newton3, iter_newton3_3 = metodo_newton(f3, df3, 1.5)
raiz3_secante3, iter_secante3_3 = metodo_secante(f3, 1, 2)

print(f"\n--- SOLUCIONES ---")
print(f"Raíz 1 (Bisección): {raiz1_bisec3:.6f} | Newton: {raiz1_newton3:.6f} | Secante: {raiz1_secante3:.6f}")
print(f"Raíz 2 (Bisección): {raiz2_bisec3:.6f} | Newton: {raiz2_newton3:.6f} | Secante: {raiz2_secante3:.6f}")
print(f"Raíz 3 (Bisección): {raiz3_bisec3:.6f} | Newton: {raiz3_newton3:.6f} | Secante: {raiz3_secante3:.6f}")

# Corridas (Se imprime solo la corrida de la Raíz 3 como ejemplo)
imprimir_corrida(iter_bisec3_3, "Raíz 3", "Bisección")
imprimir_corrida(iter_newton3_3, "Raíz 3", "Newton-Raphson")
imprimir_corrida(iter_secante3_3, "Raíz 3", "Secante")

# Graficar
raices_ej3 = [raiz1_bisec3, raiz2_bisec3, raiz3_bisec3]
graficar_funcion(f3, raices_ej3, [-2, 3], "x³ - x²e^(-0.5x) - 3x = -1 (Tres Raíces)", 3)

# =============================================
# EJERCICIO 4: cos²x - 0.5xe^(0.3x) + 5 = 0
# =============================================
print("\n" + "=" * 60)
print("EJERCICIO 4: cos²x - 0.5xe^(0.3x) + 5 = 0")
print("Intervalo de búsqueda: [3, 4]")
print("=" * 60)

def f4(x):
    return (np.cos(x))**2 - 0.5*x*np.exp(0.3*x) + 5

def df4(x):
    # Derivada: -sin(2x) - 0.5e^(0.3x) * (1 + 0.3x)
    return -np.sin(2*x) - 0.5*np.exp(0.3*x)*(1 + 0.3*x)

# Aplicar métodos al intervalo [3, 4]
a4, b4 = 3, 4

raiz_bisec4, iter_bisec4 = metodo_biseccion(f4, a4, b4)
raiz_newton4, iter_newton4 = metodo_newton(f4, df4, 3.5)
raiz_secante4, iter_secante4 = metodo_secante(f4, a4, b4)

print(f"\n--- SOLUCIONES ---")
print(f"Bisección: {raiz_bisec4:.6f}")
print(f"Newton: {raiz_newton4:.6f}")
print(f"Secante: {raiz_secante4:.6f}")

# Corridas (Tablas)
imprimir_corrida(iter_bisec4, "Raíz", "Bisección")
imprimir_corrida(iter_newton4, "Raíz", "Newton-Raphson")
imprimir_corrida(iter_secante4, "Raíz", "Secante")

# Graficar
graficar_funcion(f4, [raiz_bisec4], [0, 5], "cos²x - 0.5xe^(0.3x) + 5 = 0", 4)

# =============================================
# GRÁFICA COMPARATIVA FINAL
# =============================================
print("\n" + "=" * 60)
print("GENERANDO GRÁFICA COMPARATIVA FINAL")
print("=" * 60)

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Ejercicio 1
x1 = np.linspace(2, 4, 1000)
axs[0,0].plot(x1, f1(x1), 'b-', linewidth=2)
axs[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[0,0].plot(raiz1_bisec, f1(raiz1_bisec), 'ro', markersize=6)
axs[0,0].set_title('E1: $x^3 - e^{0.8x} = 20$')
axs[0,0].grid(True, alpha=0.3)

# Ejercicio 2
x2 = np.linspace(4.5, 7, 1000)
axs[0,1].plot(x2, f2(x2), 'b-', linewidth=2)
axs[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[0,1].plot(raiz_bisec2, f2(raiz_bisec2), 'ro', markersize=6)
axs[0,1].set_title('E2: $3\sin(0.5x) - 0.5x + 2 = 0$')
axs[0,1].grid(True, alpha=0.3)

# Ejercicio 3
x3 = np.linspace(-2, 3, 1000)
axs[1,0].plot(x3, f3(x3), 'b-', linewidth=2)
axs[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
for r in raices_ej3:
    axs[1,0].plot(r, f3(r), 'ro', markersize=6)
axs[1,0].set_title('E3: $x^3 - x^2e^{-0.5x} - 3x + 1 = 0$')
axs[1,0].grid(True, alpha=0.3)

# Ejercicio 4
x4 = np.linspace(3, 4, 1000)
axs[1,1].plot(x4, f4(x4), 'b-', linewidth=2)
axs[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1,1].plot(raiz_bisec4, f4(raiz_bisec4), 'ro', markersize=6)
axs[1,1].set_title('E4: $\cos^2x - 0.5xe^{0.3x} + 5 = 0$')
axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

