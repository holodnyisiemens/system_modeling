'''ЗАДАЧИ:
1.	Используя данные, полученные во время выполнения лабораторной работы №1, определить производные в краевых точках, численные производные 2-го порядка
2.	Используя разложение в ряд Тейлора разностную схему производной 2-го порядка с использованием 5 точек. Сравнить результаты расчета
3.	Разложить функцию в ряд Маклорена:
a.	до 1-го ненулевого значения
b.	до 2-го ненулевого значения
c.	до 3-го ненулевого значения
d.	до 4-го ненулевого значения
Построить исходный график функции совместно с разложениями в ряд, проанализировать результаты, сделать выводы.
'''

import matplotlib.pyplot as plt
import numpy as np
from math import cosh, cos, sinh, sin, factorial
import pandas as pd

# границы для формирования выборок y
LEFT_BORDER = -10
RIGTH_BORDER = 10
STEP = 0.1

# исходная функция x = cosh(cos(y))
def x_y(y):
    return cosh(cos(y))

# функциональная зависимость x от y
# принимает список значений y и возвращает список значений x
def get_x_on_y_list(y):
    x = []
    for yi in y:
        try:
            x.append(x_y(yi))
        except:
            x.append(None)
    return x

# аналитические значения производных функции y в точках y через производную x (см. теорему о дифференцировании обратной функции) 
# принимает список точек y и возвращает список производных функции в этих точках
def analyt_deriv_y(y0):
    y_deriv = []
    for y0i in y0:
        try:
            y_deriv.append(1/(-sinh(cos(y0i)) * sin(y0i)))
        except ZeroDivisionError:
            # пропуск значения при возникновении ошибки деления на ноль
            y_deriv.append(None)

    return y_deriv

# левосторонняя производная
# принимает список точек x и y и возвращает список производных функции в этих точках
def left_deriv_y(x, y):
    y_deriv = []
    
    # пропуск первого значения (т.к. для вычисления производной в точке нужны значения x и y в предыдущей точке)
    y_deriv.append(None)
    for i in range(1, len(x)):
        try:
            y_deriv.append((y[i] - y[i - 1]) / (x[i] - x[i - 1]))
        except ZeroDivisionError:
            # пропуск значения при возникновении ошибки деления на ноль
            y_deriv.append(None)

    return y_deriv

# правосторонняя производная
# принимает список точек x и y и возвращает список производных функции в этих точках
def right_deriv_y(x, y):
    y_deriv = []
    
    # пропуск первого значения (т.к. для вычисления производной в точке нужны значения x и y в следующей точке)
    for i in range(len(x) - 1):
        try:
            y_deriv.append((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
        except ZeroDivisionError:
            # пропуск значения при возникновении ошибки деления на ноль
            y_deriv.append(None)

    y_deriv.append(None)
    return y_deriv

# производные в краевых точках (для левой и правой соответственно)
def get_edge_left_deriv_y(x):
    try:
        return (2 * STEP) / (4 * x[1] - 3 * x[0] - x[2])
    except ZeroDivisionError:
        # пропуск значения при возникновении ошибки деления на ноль
        return None

def get_edge_rigth_deriv_y(x):
    try:
        return (2 * STEP) / (3 * x[-1] - 4 * x[-2] + x[-3])
    except ZeroDivisionError:
        # пропуск значения при возникновении ошибки деления на ноль
        return None

# вторая производная по трем точкам
def second_num_derivs_y_by_3_dots(x):
    sec_derivs_y = []
    sec_derivs_y.append(None)

    # пропуск первого и последнего значений
    for i in range(1, len(x) - 1):
        try:
            sec_derivs_y.append(pow(STEP, 2) / (x[i + 1] - 2 * x[i] + x[i - 1]))
        except ZeroDivisionError:
            # пропуск значения при возникновении ошибки деления на ноль
            sec_derivs_y.append(None)

    sec_derivs_y.append(None)
    return sec_derivs_y

# вторая производная по пяти точкам
def second_num_derivs_y_by_5_dots(x):
    sec_derivs_y = []
    sec_derivs_y.append(None)
    sec_derivs_y.append(None)

    # пропуск первого и последнего значений
    for i in range(2, len(x) - 2):
        try:
            sec_derivs_y.append(
                (pow(STEP, 2)) / 
                (x[i - 2] - 3 * x[i - 1] + 4 * x[i] - 3 * x[i + 1] + x[i + 2])
            )
        except ZeroDivisionError:
            # пропуск значения при возникновении ошибки деления на ноль
            sec_derivs_y.append(None)

    sec_derivs_y.append(None)
    sec_derivs_y.append(None)
    return sec_derivs_y

# значения функции для ряда Маклорена (до 1-го ненулевого значения)
def first_non_zero_values_maclaurin(y):
    return [1 for _ in range(len(y))]

# значения функции для ряда Маклорена (до 2-го ненулевого значения)
def second_non_zero_values_maclaurin(y):
    first_vals = first_non_zero_values_maclaurin(y)
    values = []
    for i in range(len(y)):
        values.append(first_vals[i] + pow(cos(y[i]), 2) / 2)
    return values

# значения функции для ряда Маклорена (до 3-го ненулевого значения)
def third_non_zero_values_maclaurin(y):
    second_vals = second_non_zero_values_maclaurin(y)
    values = []
    for i in range(len(y)):
        values.append(second_vals[i] + pow(cos(y[i]), 4) / factorial(4))
    return values

# значения функции для ряда Маклорена (до 4-го ненулевого значения)
def forth_non_zero_values_maclaurin(y):
    third_vals = third_non_zero_values_maclaurin(y)
    values = []
    for i in range(len(y)):
        values.append(third_vals[i] + pow(cos(y[i]), 6) / factorial(6))
    return values

# формирование выборки y и расчет по ним производных
y1 = [y for y in np.arange(LEFT_BORDER, RIGTH_BORDER + STEP, STEP)]
analyt_y1 = analyt_deriv_y(y1)
x1 = get_x_on_y_list(y1)
left_y1 = left_deriv_y(x1, y1)
right_y1 = right_deriv_y(x1, y1)

# изменение пустых значений в краевых точках
left_y1[0] = get_edge_left_deriv_y(x1)
right_y1[-1] = get_edge_rigth_deriv_y(x1)

# вторые производные численными методами
sec_derivs_y1_3_dots = second_num_derivs_y_by_3_dots(x1)
sec_derivs_y1_5_dots = second_num_derivs_y_by_5_dots(x1)

# таблица x, y и производных y
data1 = []
for i in range(len(x1)):
    data1.append((x1[i], y1[i], analyt_y1[i], left_y1[i], right_y1[i], sec_derivs_y1_3_dots[i], sec_derivs_y1_5_dots[i]))

table1 = pd.DataFrame(data1, columns=['x', 'y', "y'аналит", "y'лев", "y'прав", "y'' по 3 т.", "y'' по 5 т."])
table1.reset_index(drop=True)
print(table1)

# таблица исходных значений x от y и полученных при помощи разложения в ряд Маклорена
data2 = []
fst = first_non_zero_values_maclaurin(y1)
sec = second_non_zero_values_maclaurin(y1)
thrd = third_non_zero_values_maclaurin(y1)
frth = forth_non_zero_values_maclaurin(y1)
for i in range(len(x1)):
    data2.append((x1[i], fst[i], sec[i], thrd[i], frth[i]))

table2 = pd.DataFrame(data2, columns=['true x', 'Maclaurin 1st != 0', 'Maclaurin 2nd != 0', 'Maclaurin 3rd != 0', 'Maclaurin 4th != 0'])
print(table2)

# исходный график функции совместно с разложениями в ряд Маклорена
plt.plot(x1, y1, color='b', label='original')
plt.plot(fst, y1, color='y', label='1st!=0', alpha=0.9)
plt.plot(sec, y1, color='g', label='2nd!=0', alpha=0.8)
plt.plot(thrd, y1, color='m', label='3rd!=0', alpha=0.7)
plt.plot(frth, y1, color='r', label='4rd!=0', alpha=0.5)
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.10), ncol=5)
plt.title('Исходный график функции совместно с разложениями в ряд')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
