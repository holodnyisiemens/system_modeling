'''ЗАДАЧИ:
1.	Построить график функцию x=ch⁡(cos⁡(y)) на интервале y=[-10, 10]
2.	Найти аналитическое решение производной y'(x), численное решение левосторонней, правосторонней и центральных производных, представить их на графике функции.
3.	На этом же диапазоне провести вычисления всех производных при других (пяти) шагах. Построить зависимости погрешностей вычисления центральных производных от координаты.
'''

import matplotlib.pyplot as plt
import numpy as np
from math import cosh, cos, sinh, sin

# границы для формирования выборок y
LEFT_BORDER = -10
RIGTH_BORDER = 10

# функциональная зависимость x от y
# принимает список значений y и возвращает список значений x
def x_on_y(y):
    x = []
    for yi in y:
        x.append(cosh(cos(yi)))
    return x

# аналитические значения производных фукнции y в точках y через производную x (см. теорему о дифференцировании обратной функции) 
# принимает список точек y и возвращает список производных функции в этих точках
def analyt_deriv_y(y0):
    y_deriv = []
    for y0i in y0:
        analyt_deriv_x = -sinh(cos(y0i))*sin(y0i)

        # пропуск значений при знаменателе равному 0
        if analyt_deriv_x != 0:
            y_deriv.append(1/analyt_deriv_x)
        else:
            y_deriv.append(None)

    return y_deriv

# левосторонняя производная
# принимает список точек x и y и возвращает список производных функции в этих точках
def left_deriv_y(x, y):
    y_deriv = []
    
    # пропуск первого значения (т.к. для вычисления производной в точке нужны значения x и y в предыдущей точке)
    y_deriv.append(None)
    for i in range(1, len(x)):
        denominator = x[i]-x[i-1]

        # пропуск значений при знаменателе равному 0
        if denominator != 0:
            y_deriv.append((y[i]-y[i-1])/denominator)
        else:
            y_deriv.append(None)

    return y_deriv

# правосторонняя производная
# принимает список точек x и y и возвращает список производных функции в этих точках
def right_deriv_y(x, y):
    y_deriv = []
    
    # пропуск первого значения (т.к. для вычисления производной в точке нужны значения x и y в следующей точке)
    for i in range(len(x) - 1):
        denominator = x[i+1]-x[i]

        # пропуск значений при знаменателе равному 0
        if denominator != 0:
            y_deriv.append((y[i+1]-y[i])/denominator)
        else:
            y_deriv.append(None)

    y_deriv.append(None)
    return y_deriv

# центральная производная
# принимает список точек x и y и возвращает список производных функции в этих точках
def central_deriv_y(x, y):
    y_deriv = []
    
    # пропуск первого и последнего значений (т.к. для вычисления производной в точке нужны значения x и y в предыдущей и следующей точках)
    y_deriv.append(None)

    for i in range(1, len(x) - 1):
        denominator = x[i+1]-x[i-1]
                       
        # пропуск значений при знаменателе равному 0
        if denominator != 0:
            y_deriv.append((y[i+1]-y[i-1])/denominator)
        else:
            y_deriv.append(None)

    y_deriv.append(None)
    return y_deriv

# получение трех столбцов данных по выборке: левосторонняя, правосторонняя и центральная производные
def get_numerical_derivatives(x, y):
    left = left_deriv_y(x, y)
    right = right_deriv_y(x, y)
    central = central_deriv_y(x, y)
    return left, right, central

# получение списка абсолютных погрешностей по спискам правильных значений и рассчитанных
def get_absolute_inaccuracy(real, calculated):
    abs_inac = []
    abs_inac.append(None)
    for i in range(1, len(real) - 1):
        if real[i] and calculated[i]:
            abs_inac.append(abs(real[i] - calculated[i]))
        else:
            abs_inac.append(None)

    abs_inac.append(None)
    return abs_inac

# формирование выборки y с шагом 1 и расчет по ним производных
y1 = [y for y in range(LEFT_BORDER, RIGTH_BORDER + 1, 1)]
analyt_y1 = analyt_deriv_y(y1)
x1 = x_on_y(y1)
left_y1, right_y1, central_y1 = get_numerical_derivatives(x1, y1)

# построение графика исходной функции
plt.plot(x1, y1)
plt.title('Исходная функция')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

# построение графиков для отображения расхождения между аналитической и численными производными 
plt.scatter(x1, analyt_y1, color='r', label='Аналитич.')
plt.scatter(x1, left_y1, color='y', label='Лев.')
plt.scatter(x1, right_y1, color='g', label='Прав.')
plt.scatter(x1, central_y1, color='b', label='Центр.')
plt.title('Графики аналитической и численных производных\n\n')
plt.xlabel('x')
plt.ylabel("y'(x)")
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=4)
plt.grid()
plt.show()

# уменьшение шага
steps = [0.5, 0.25, 0.125, 0.1, 0.05]

# цвета для графика
colors = ['b', 'y', 'g', 'r', 'm']

# формирование выборок y с шагами из steps и расчет по ним численных производных
# и абсолютных погрешностей между центральной производной и аналитическим значениям
for step, color in zip(steps, colors):
    # округление, т.к. при приведении в двоичный вид число будет терять точность
    y2 = [round(y, 3) for y in np.arange(LEFT_BORDER, RIGTH_BORDER + step, step)]
    analyt_y2 = analyt_deriv_y(y2)
    x2 = x_on_y(y2)
    left_y2, right_y2, central_y2 = get_numerical_derivatives(x2, y2)
    abs_inac_2 = get_absolute_inaccuracy(analyt_y2, central_y2)

    # построение графиков для оценки погрешности при уменьшении шага при расчете численных производных 
    plt.scatter(x2, abs_inac_2, color=color, label=f'шаг:{step}', s=5)

plt.title('Зависимость погрешности расчета центральных производных от x\n\n')
plt.xlabel('x')
plt.ylabel("Погрешность")
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=5)
plt.grid()
plt.show()
