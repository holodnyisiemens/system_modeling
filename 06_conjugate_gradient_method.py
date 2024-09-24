from math import sin, cos, e, sqrt, pi
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

opt_x_data = []
opt_y_data = []
opt_z_data = []

# экспонента
def exp(num):
    # при переполнении возвращается бесконечность
    try:
        return e ** num
    except OverflowError:
        return float('inf')

# исходная функция, которую требуется оптимизировать
def f(x, z):
    # отброс периодической части
    return sin(z % (2*pi)) + exp(-x)

# норма градиента
def get_gradient_norm(x_grad, z_grad):
    return sqrt(x_grad ** 2 + z_grad ** 2)

# первая частная производная y по x
def get_first_part_deriv_y_on_x(x):
    return -exp(-x)

# первая частная производная y по z
def get_first_part_deriv_y_on_z(z):
    # отброс периодической части
    return cos(z % (2*pi))

# получение градиента (вектора частных производных)
def get_gradient(x, z):
    return get_first_part_deriv_y_on_x(x), get_first_part_deriv_y_on_z(z)

# поправочный коэффициент w
def get_correction_coeff(x, z, x_old, z_old):
    x_grad, z_grad = get_gradient(x, z)
    x_grad_old, z_grad_old = get_gradient(x_old, z_old)
    norm_x_grad, norm_z_grad = get_gradient_norm(x_grad, z_grad)
    norm_x_grad_old, norm_z_grad_old = get_gradient_norm(x_grad_old, z_grad_old)
    w_x = (norm_x_grad / norm_x_grad_old) ** 2
    w_z = (norm_z_grad / norm_z_grad_old) ** 2
    return w_x, w_z

# сопряженный вектор S
def get_s(x, z, x_old, z_old, s_old_x, s_old_z):
    # поправочные коэффициенты w
    w_x, w_z = get_correction_coeff(x, z, x_old, z_old)
    new_s_x = - get_gradient(x, z) + w_x * s_old_x
    new_s_z = - get_gradient(x, z) + w_z * s_old_z
    return new_s_x, new_s_z

def optimize(lamb, x, z, s_old_x=None, s_old_z=None, acc=None):
    global opt_x_data, opt_y_data, opt_z_data
    opt_x_data.append(x)
    opt_y_data.append(f(x, z))
    opt_z_data.append(z)

    x_grad, z_grad = get_gradient(x, z)
    try:
        # проверка нормы градиента
        if acc and (get_gradient_norm(x_grad, z_grad) < acc):
            return x, z
    except:
        # при возникновении ошибок вычисления нормы градиента
        return x, z

    if not (s_old_x and s_old_z):
        # начальный сопряженный вектор
        s_old_x = - x_grad
        s_old_z = - z_grad

    x += lamb * s_old_x
    z += lamb * s_old_z

    try:
        return optimize(lamb, x, z, s_old_x, s_old_z, acc)
    except RecursionError:
        return x, z

# шаг сходимости
lamb = 0.001

# 30 тестов 
x_data = [random.randint(-100, 100) for _ in range(31)]
z_data = [random.randint(-100, 100) for _ in range(31)]

# таблица с результатами
data = []

for x_start, z_start in zip(x_data, z_data):
    # начальное значение функции
    y_start = f(x_start, z_start)

    # значения после оптимизации
    opt_x, opt_z = optimize(lamb, x_start, z_start)
    opt_y = f(opt_x, opt_z)

    # проверка на уменьшение найденного минимума функции и соответствие с ожидаемым результатом (-1)
    if opt_y < -0.9 and opt_y < y_start:
        result = 'PASS'
    else:
        result = 'FAIL'

    # слишком большие значения y будут отображаться как бесконечность
    if y_start > 1_000_000_000:
        y_start = float('inf')
    if opt_y > 1_000_000_000:
        opt_y = float('inf')
    if opt_x > 1_000_000_000:
        opt_x = float('inf')

    data.append((x_start, z_start, y_start, opt_x, opt_z, opt_y, result))

table = pd.DataFrame(data, columns=['x_start', 'z_start', 'y_start', 'opt_x', 'opt_z', 'opt_y', 'correctness'])
print(table)
print('PASSED:', len(table[table['correctness']=='PASS']))

opt_x_data = []
opt_y_data = []
opt_z_data = []

# шаг сходимости и исходные значения
lamb = 0.01
x_start = 0
z_start = 3.9

# оптимизация (включает запись данных точек траектории спуска)
opt_x, opt_z = optimize(lamb, x_start, z_start)
print(opt_x, opt_z, f(opt_x, opt_z))

# данные для осей
x = np.linspace(-3, 15, 100)
z = np.linspace(-3, 15, 100)
x, z = np.meshgrid(x, z)
y = np.sin(z) + np.exp(-x)

# построение графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, z, y, cmap='viridis', alpha=0.7)

# названия осей
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')

# построение траектории
ax.plot(opt_x_data, opt_z_data, opt_y_data, color='red', linewidth=3)

# отображение графика
plt.show()
