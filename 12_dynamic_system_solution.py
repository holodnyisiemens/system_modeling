from math import cosh, sinh
import numpy as np
import matplotlib.pyplot as plt

# аналитическое решение
def x(t):
    return np.cosh(t)

# скорость
def v(t):
    return np.sinh(t)

# ускорение
def a(t):
    return np.cosh(t)

# метод Leapfrog
def leapfrog_method(x0, v0, a0, t_start, t_end, h):
    num_iter = int((t_end - t_start) / h)

    # вектор значений искомой функции
    X = np.empty(shape=num_iter)
    X[0] = x0

    xi = x0
    vi = v0 + a0 * h / 2
    ti = t_start

    for i in range(1, num_iter):
        ti += h

        # аналитическое значение ускорения (исходная функция)
        ai = a(ti)

        vi += ai * h
        xi += vi * h

        X[i] = xi
    return X

# Метод Стёрмера-Верле
def stermer_werle_method(x0, x1, t_start, t_end, h):
    num_iter = int((t_end - t_start) / h)

    X = np.empty(shape=num_iter)

    ti = t_start
    X[0] = x0

    ti += h
    X[1] = x1

    for i in range(1, num_iter - 1):
        ti += h
        ai = a(ti)
        X[i + 1] = 2 * X[i] - X[i - 1] + ai * (h ** 2)
    
    return X

# Метод Рунге-Кутты IV порядка
def Runge_Kutta_method_4th_order(x0, v0, t_start, t_end, h):
    num_iter = int((t_end - t_start) / h)

    X = np.empty(shape=num_iter)
    V = np.empty(shape=num_iter)

    X[0] = x0
    V[0] = v0

    for i in range(num_iter - 1):
        x1 = X[i] + V[i] * h / 2
        v1 = V[i] + X[i] * h / 2

        x2 = X[i] + v1 * h / 2
        v2 = V[i] + x1 * h / 2

        x_pred = X[i] + v2 * h
        v_pred = V[i] + x2 * h

        X[i + 1] = X[i] + (V[i] + 2 * (v1 + v2) + v_pred) * h / 6
        V[i + 1] = V[i] + (X[i] + 2 * (x1 + x2) + x_pred) * h / 6

    return X, V

# границы времени и шаг
t_start = -2
t_end = 2
h = 0.001

# известные значения
x0 = x(t_start)

x1 = x(t_start + h)

v0 = v(t_start)
a0 = a(t_start)

X_leapfrog = leapfrog_method(x0, v0, a0, t_start, t_end, h)
X_stremer_werle = stermer_werle_method(x0, x1, t_start, t_end, h)
X_runge_kutta, V_runge_kutta = Runge_Kutta_method_4th_order(x0, v0, t_start, t_end, h)

T = np.arange(t_start, t_end, h)
X_true = x(T)
V_true = v(T)

# графики восстановленных функций
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.suptitle('Графики восстановленных функций')

axs[0, 0].plot(T, X_leapfrog, label='М. Leapfrog', linewidth=2)
axs[0, 1].plot(T, X_stremer_werle, label='М. Стёрмера-Верле', linewidth=2)
axs[1, 0].plot(T, X_runge_kutta, label='М. Рунге-Кутты: функция', linewidth=2)

axs[1, 1].plot(T, V_true, label='Функция скорости (аналит.)', linewidth=4)
axs[1, 1].plot(T, V_runge_kutta, label='М. Рунге-Кутты: скорость', linewidth=2)

for i in range(2):
    for j in range(2):
        if not (i == 1 and j == 1):
            axs[i, j].plot(T, X_true, label='Искомая функция', linewidth=4, zorder=0)
        axs[i, j].legend(loc='best', fontsize=8)

# настройка расстояния между подграфиками
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()
