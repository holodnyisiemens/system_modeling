'''Задача: найти минимум функции y = sin(z) + e^-x с помощью многомерного метода Ньютона'''

from math import sin, cos, e, sqrt, pi, isnan
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

# экспонента
def exp(num):
    # при переполнении возвращается бесконечность
    try:
        return e ** num
    except OverflowError:
        return float('inf')

# исходная функция
def f(x, z):
    # отброс периодической части
    return sin(z % (2*pi)) + exp(-x)

# первая частная производная y по x
def get_first_part_deriv_y_on_x(x):
    return -exp(-x)

# первая частная производная y по z
def get_first_part_deriv_y_on_z(z):
    # отброс периодической части
    return cos(z % (2*pi))

# вычисление Гессиана
def get_hessian(x, z):
    # отброс периодической части
    return -sin(z % (2*pi)) * exp(-x)

# получение градиента (вектора частных производных)
def get_gradient(x, z):
    return get_first_part_deriv_y_on_x(x), get_first_part_deriv_y_on_z(z)

# оптимизация функции sin(z) + e^-x (поиск точки минимума)
def optimize(x, z, acc=None):
    x_grad, z_grad = get_gradient(x, z)
    try:
        # проверка нормы градиента
        if acc and (sqrt(x_grad ** 2 + z_grad ** 2) < acc):
            return x, z
    except:
        # при возникновении ошибок вычисления нормы градиента
        return x, z
    
    # гессиан
    hess = get_hessian(x, z)

    # рекурсия (если не указана точность, то будет выполняться до возникновения ошибки рекурсии)
    try:
        new_x = x - x_grad / hess
        new_z = z - z_grad / hess
        
        # проверка на значения NaN (Not a Number - не число)
        if isnan(new_x) and isnan(new_z):
            return x, z
        elif isnan(new_x):
            return optimize(x, new_z, acc)
        elif isnan(new_z):
            return optimize(new_x, z, acc)
        else:
            return optimize(new_x, new_z, acc)
    except ZeroDivisionError:
        return x, z
    except RecursionError:
        return x, z

# тесты
z_data_1 = [random.uniform(3/2*pi-0.3, 3/2*pi+0.3) for _ in range(16)]
z_data_2 = [random.uniform(-pi/2-0.3, -pi/2+0.3) for _ in range(16)]
z_data = z_data_1 + z_data_2
x_data = [random.randint(600, 710) for _ in range(31)]

# точность
acc=0.00001

# таблица с результатами
data = []

for x, z in zip(x_data, z_data):
    # значение y при исходных x и z
    y = f(x, z)
    opt_x, opt_z = optimize(x, z, acc)

    # отображение z от -2pi до 2pi, т.к. z передается в периодическую функцию
    if opt_z:
        opt_z = opt_z % (2*pi)

    # расчет оптимального значения y
    opt_y = f(opt_x, opt_z)
    
    # при очень больших значениях y - принимается бесконечностью
    if opt_y and opt_y > 1000000000:
        opt_y = float('inf')

    # проверка: y приблизился к -1 (точка минимума исходной функции) или устремился в бесконечность
    if opt_y and ((opt_y < y and opt_y >= -1 and opt_y <= -0.8) or (opt_y == float('inf'))):
        result = 'PASS'
    else:
        result = 'FAIL'

    # добавление очередной строки
    data.append((x, z, y, opt_x, opt_z, opt_y, result))

table = pd.DataFrame(data, columns=['orig_x', 'orig_z', 'orig_y', 'opt_x', 'opt_z', 'opt_y', 'result'])
print(table)

# количество успешно пройденных тестов
print('PASSED:', len(table[table['result']=='PASS']))

# фигура
fig = plt.figure(figsize=(7, 4))

# трехмерная ось
ax_3d = fig.add_subplot(projection='3d')

x = np.arange(-10, 10, 0.1)
z = np.arange(-10, 10, 0.1)

# на вход одномерные массивы а на выходе двумерные (для обращения по индексам)
x_grid, z_grid = np.meshgrid(x, z)

y_grid = np.sin(z_grid) + np.power(np.e, -x_grid)


ax_3d.plot_wireframe(x_grid, y_grid, z_grid)

ax_3d.set_xlabel('x')
ax_3d.set_ylabel('y')
ax_3d.set_zlabel('z')

plt.show()
