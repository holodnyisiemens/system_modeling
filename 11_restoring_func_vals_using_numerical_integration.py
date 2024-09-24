import numpy as np
import matplotlib.pyplot as plt

# функция, которую нужно интегрировать
def f(x):
    return (x ** 5 - x + 1) / (x ** 2 + 1)

# аналитическое значение интеграла функции
def analyt_integral(x):
    return np.arctan(x) + x ** 4 / 4 - x ** 2 / 2

# метод Эйлера при известном начальном/конечном значении
def Euler_method(x_know, y_know, h, n, knowing_value):
    # x0, y0 - начальные условия
    # h - шаг = xi+1 - xi
    # n - количество шагов
    Y = np.empty(shape=(n,))
    if knowing_value == 'start':
        Y[0] = y_know
        xi = x_know
        yi = y_know
        for i in range(1, n):
            yi += f(xi) * h
            xi += h
            Y[i] = yi
    elif knowing_value == 'end':
        Y[0] = y_know
        xi = x_know
        yi = y_know
        for i in range(1, n):
            yi -= f(xi) * h
            xi -= h
            Y[i] = yi
        return Y[::-1]
    else:
        raise ValueError("Incorrect value for argument 'knowing_value'")

    return Y

# модифицированный метод Эйлера с пересчетом
def modified_Euler_method(x0, y0, h, n):
    # x0, y0 - начальные условия
    # h - шаг = xi+1 - xi
    # n - количество шагов
    Y = np.empty(shape=(n,))
    Y[0] = y0
    xi = x0
    yi = y0

    for i in range(1, n):
        # не используется, т.к. функция не зависит от y
        # predict_yi = yi + f(xi) * h
        corrected_yi = yi + (f(xi) + f(xi + h)) / 2 * h
        yi = corrected_yi
        xi += h
        Y[i] = yi

    return Y

# двухшаговый метод Адамса-Бошфорта
def two_step_Adams_Boschfort_method(x0, y0, h, n):
    # x0, y0 - начальные условия
    # h - шаг = xi+1 - xi
    # n - количество шагов
    Y = np.empty(shape=(n,))
    Y[0] = y0
    xi = x0
    yi = y0

    for i in range(1, n):
        x_prev = xi
        xi += h
        yi += 3/2 * h * f(xi) - 1/2 * h * f(x_prev)
        Y[i] = yi

    return Y

# Метод Рунге-Кутты IV порядка
def Runge_Kutta_method_4th_order(x0, y0, h, n):
    # x0, y0 - начальные условия
    # h - шаг = xi+1 - xi
    # n - количество шагов
    Y = np.empty(shape=(n,))
    Y[0] = y0
    xi = x0
    yi = y0

    for i in range(1, n):
        xi += h
        k1 = f(xi)
        k2 = f(xi + h / 2)
        k3 = f(xi + h / 2)
        k4 = f(xi + h)
        yi += h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i] = yi

    return Y

# получение значений относительных погрешностей
def get_abs_errs(Y_approx, Y_true):
    return np.abs(Y_approx - Y_true) / np.abs(Y_approx)

# начальные условия
x_start = -5
x_end = 5
y_start = analyt_integral(x_start)
y_end = analyt_integral(x_end)

# шаг и число итераций
h = 0.01
n = int((x_end - x_start) / h)

X = np.arange(x_start, x_end, h)
Y_true = analyt_integral(X)
Y_euler_start = Euler_method(x_start, y_start, h, n, knowing_value='start')
Y_euler_end = Euler_method(x_end, y_end, h, n, knowing_value='end')
Y_euler_modified = modified_Euler_method(x_start, y_start, h, n)
Y_adams_boschfort = two_step_Adams_Boschfort_method(x_start, y_start, h, n)
Y_runge_kutta = Runge_Kutta_method_4th_order(x_start, y_start, h, n)

# графики восстановленных функций
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
fig.suptitle('Графики восстановленных функций')

axs[0, 0].plot(X, Y_true, label='Искомая функция', linewidth=2, color='orange')
axs[0, 0].legend(loc='best', fontsize=8)

axs[0, 1].plot(X, Y_euler_start, label='м. Эйлера (нач. зн-я)', linewidth=2)
axs[1, 0].plot(X, Y_euler_end, label='м. Эйлера (конеч. зн-я)', linewidth=2)
axs[1, 1].plot(X, Y_euler_modified, label='Модиф. м. Эйлера', linewidth=2)
axs[2, 0].plot(X, Y_adams_boschfort, label='м. Адамса-Бошфорта', linewidth=2)
axs[2, 1].plot(X, Y_runge_kutta, label='м. Рунге-Кутты', linewidth=2)

for i in range(3):
    for j in range(2):
        if not (i == 0 and j == 0):
            axs[i, j].plot(X, Y_true, label='Искомая функция', linewidth=4, zorder=0)
            axs[i, j].legend(loc='best', fontsize=8)

# настройка расстояния между подграфиками
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()

# графики относительных погрешностей
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
fig.suptitle('Графики относительных погрешностей')

axs[0, 0].plot(X, Y_true, label='Искомая функция', linewidth=2, color='orange')
axs[0, 0].legend(loc='best', fontsize=8)

axs[0, 1].plot(X, get_abs_errs(Y_euler_start, Y_true), label='м. Эйлера (нач. зн-я)', linewidth=2)
axs[1, 0].plot(X, get_abs_errs(Y_euler_end, Y_true), label='м. Эйлера (конеч. зн-я)', linewidth=2)
axs[1, 1].plot(X, get_abs_errs(Y_euler_modified, Y_true), label='Модиф. м. Эйлера', linewidth=2)
axs[2, 0].plot(X, get_abs_errs(Y_adams_boschfort, Y_true), label='м. Адамса-Бошфорта', linewidth=2)
axs[2, 1].plot(X, get_abs_errs(Y_runge_kutta, Y_true), label='м. Рунге-Кутты', linewidth=2)

for i in range(3):
    for j in range(2):
        if not (i == 0 and j == 0):
            axs[i, j].legend(loc='best', fontsize=8)

# Настройка расстояния между подграфиками
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()
