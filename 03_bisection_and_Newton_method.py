'''Свести график функции x=ch⁡cos⁡y  к нелинейным уравнениям типа f(y) = 0. 
Найти четыре корня с применением методов бисекции, Ньютона
и его частных случаев (метода секущей и метода одной касательной).'''

import math
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# исходная функция x = cosh(cos(y)) - 1.39
def x_y(y):
    return math.cosh(math.cos(y)) - 1.39

# производная исходной функции
def deriv_x_on_y(y):
    return -math.sinh(math.cos(y)) * math.sin(y)

# рекурсивная функция поиска одного из корней уравнения методом бисекции
# вычисляется по промежутку и точности
def bisection_method(ymin, ymax, e=None):
    # середина отрезка
    ymid = (ymin + ymax) / 2

    # если указана точность, то проверка точности
    if e and abs(x_y(ymid)) < e:
        return ymid

    # c какой стороны от f(ymid) наблюдается смена знака функции – в той половине наблюдается пересечение с осью OY
    # вычислить смену знака можно умножением значений функций
    if x_y(ymin) * x_y(ymid) < 0:
        ymax = ymid
    elif x_y(ymid) * x_y(ymax) < 0:
        ymin = ymid
    else:
        return ymid

    # если точность не указана, рекурсия продолжается до возникновения исключения
    try:
        # рекурсивный случай
        return bisection_method(ymin, ymax, e)
    except:
        return ymid

# одномерный метод Ньютона по приблизительному значению и точности
def newton_method(approx_y, e=None):
    if e and abs(x_y(approx_y)) < e:
        return approx_y
    try:
        approx_y = approx_y - (x_y(approx_y) / deriv_x_on_y(approx_y))
        return newton_method(approx_y, e)
    except:
        return approx_y

# метод секущих (частный случай одномерного метода Ньютона)
# выбираются две точки вокруг приблизительного решения
# каждая последующая итерация, зависящая от двух предыдущих является уточненным решением
def secant_method(y_prev, y_n, e=None):
    if e and abs(x_y(y_n)) < e:
        return y_n
    try:
        new_y_prev = y_n
        y_n = y_n - x_y(y_n) * (y_n - y_prev) / (x_y(y_n) - x_y(y_prev))
        return secant_method(new_y_prev, y_n, e)
    except:
        return y_n

# метод одной касательной (частный случай одномерного метода Ньютона)
# в итерационном подходе используется значения производной только при изначальной точке отсчета 
def single_tangent_method(y_0, y_n, e=None):
    if e and abs(x_y(y_n)) < e:
        return y_n
    try:
        y_n = y_n - (x_y(y_n) / deriv_x_on_y(y_0))
        return single_tangent_method(y_0, y_n, e)
    except:
        return y_n

# все участки, где есть ровно 1 точка y, обнуляющая функцию x от y
borders = [(-10, -9.5), (-9, -8.5), (-7, -6.5), (-6, -5.5), (-4, -3.5), (-3, -2.5), (-1, 0), (0, 1), (2.5, 3), (3.5, 4), (5.5, 6), (6.5, 7), (8.5, 9), (9.5, 10)]

# все приблизительные значения корней на каждом участке
y_approxes = [-9.8, -8.9, -6.7, -5.7, -3.5, -2.6, -0.4, 0.5, 2.7, 3.7, 5.8, 6.8, 8.9, 9.8]

# точность
e = 0.0000001

# список строк таблицы с результатами вычислений корней
data = []

random_indexes = random.sample([ind for ind in range(len(borders))], 4)
random_borders = [borders[ind] for ind in random_indexes]
random_y_approx = [y_approxes[ind] for ind in random_indexes]

for bdrs, y_approx in zip(random_borders, random_y_approx):
    res_bisection_method = bisection_method(*bdrs, e)
    res_newton_method = newton_method(y_approx, e)
    res_secant_method = secant_method(*bdrs, e)
    res_single_tangent_method = single_tangent_method(*bdrs, e)
    data.append((res_bisection_method, res_newton_method, res_secant_method, res_single_tangent_method))

table = pd.DataFrame(data, columns=['bisection_method', 'newton_method', 'secant_method', 'single_tangent'], index=random_borders)
print(table)

# формирование выборки y с шагом 0.1 и расчет по ним производных
y = [yi for yi in np.arange(-10, 11, 0.1)]
x = [x_y(yi) for yi in y]

# построение графика исходной функции
plt.plot(x, y)
plt.title('Исходная функция')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
