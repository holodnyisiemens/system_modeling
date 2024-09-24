'''На некотором диапазоне провести численное интегрирование функции методами прямоугольников (левых, правых и средних), 
трапеций и Симпсона. Построить графики изменения значения интеграла.'''

from math import atan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

left_squares = []
right_squares = []
middle_squares = []
trapezoid_squares = []
simpson_squares = []

# исходная функция, которую нужно интегрировать
def f(x: float) -> float:
    return (x ** 5 - x + 1) / (x ** 2 + 1)

# первообразная исходной функции
def antiderivative(x: float) -> float:
    return atan(x) + x ** 4 / 4 - x ** 2 / 2

# аналитическое значение интеграла исходной функции
def analytical_integration(x_min: float, x_max: float) -> float:
    return antiderivative(x_max) - antiderivative(x_min)

# метод левых прямоугольников
def left_rectangle_integration(x_min: float, x_max: float, step: float) -> float:
    global left_squares
    s = 0
    for x in np.arange(x_min, x_max, step):
        left_squares.append(s)
        s += f(x) * step
    return s

# метод правых прямоугольников    
def right_rectangle_integration(x_min: float, x_max: float, step: float) -> float:
    global right_squares
    s = 0
    for x in np.arange(x_min, x_max, step):
        right_squares.append(s)
        s += f(x + step) * step
    return s

# метод средних прямоугольников
def middle_rectangle_integration(x_min: float, x_max: float, step: float) -> float:
    global middle_squares
    s = 0
    for x in np.arange(x_min, x_max, step):
        middle_squares.append(s)
        x_middle = (x + x + step) / 2
        s += f(x_middle) * step
    return s

# метод трапеций
def trapezoid_integration(x_min: float, x_max: float, step: float) -> float:
    global trapezoid_squares
    s = 0
    for x in np.arange(x_min, x_max, step):
        trapezoid_squares.append(s)
        bases_half_sum = (f(x) + f(x + step)) / 2
        s += bases_half_sum * step
    return s

# метод Симпсона
def simpson_integration(x_min: float, x_max: float, step: float) -> float:
    global simpson_squares
    s = 0
    for x in np.arange(x_min, x_max, step):
        simpson_squares.append(s)
        s += step / 6 * (f(x) + 4 * f((x + x + step) / 2) + f(x + step))
    return s

# тесты для разных диапазонов и шагов
# функция монотонно возрастает на промежутке от 0 до 70
borders = [(0, 5), (0, 10), (0, 30), (0, 50)]
steps = [1, 0.1, 0.01, 0.001, 0.0001]
borders_and_steps = [(*bords, step) for bords in borders for step in steps]

# таблица с результатами
data = []

for x_min, x_max, step in borders_and_steps:
    # значения интеграла
    s_analyt = analytical_integration(x_min, x_max)
    s_left = left_rectangle_integration(x_min, x_max, step)
    s_right = right_rectangle_integration(x_min, x_max, step)
    s_mid = middle_rectangle_integration(x_min, x_max, step)
    s_trap = trapezoid_integration(x_min, x_max, step)
    s_sim = simpson_integration(x_min, x_max, step)

    results = map(lambda x: round(x, 3), (s_analyt, s_left, s_right, s_mid, s_trap, s_sim))

    # строка таблицы (для читаемости  оптимизированный y округляется до 4х знаков после запятой)
    data.append((x_min, x_max, step, *results))

table = pd.DataFrame(data, columns=['x_min', 'x_max', 'step', 's_analyt', 's_left', 's_right', 's_mid', 's_trap', 's_sim'])
print(table)

# относительные погрешности в долях
table['left_err'] = abs(table['s_left'] - table['s_analyt']) / abs(table['s_left'])
table['rigth_err'] = abs(table['s_right'] - table['s_analyt']) / abs(table['s_right'])
table['mid_err'] = abs(table['s_mid'] - table['s_analyt']) / abs(table['s_mid'])
table['trap_err'] = abs(table['s_trap'] - table['s_analyt']) / abs(table['s_trap'])
table['sim_err'] = abs(table['s_sim'] - table['s_analyt']) / abs(table['s_sim'])

# если отностильная погрешность меньше процента, интеграл считается верно рассчитанным
accuracy = 0.01
print('LEFT PASSED:', (table['left_err'] < accuracy).sum())
print('RIGHT PASSED:', (table['rigth_err'] < accuracy).sum())
print('MIDDLE PASSED:', (table['mid_err'] < accuracy).sum())
print('TRAPEZOID PASSED:', (table['trap_err'] < accuracy).sum())
print('SIMPSON PASSED:', (table['sim_err'] < accuracy).sum())

# построение графиков изменения площадей для каждого из методов
x_min = 0
x_max = 5
steps_and_colors = zip((0.1, 0.01, 0.001), ('m', 'y', 'r'))
steps_and_colors = list(steps_and_colors)

for step, color in steps_and_colors:
    left_squares = []
    s_left = left_rectangle_integration(x_min, x_max, step)
    plt.plot(range(len(left_squares)), left_squares, c=color)
plt.show()

for step, color in steps_and_colors:
    right_squares = []
    s_right = right_rectangle_integration(x_min, x_max, step)
    plt.plot(range(len(right_squares)), right_squares, c=color)
plt.show()

for step, color in steps_and_colors:
    middle_squares = []
    s_mid = middle_rectangle_integration(x_min, x_max, step)
    plt.plot(range(len(middle_squares)), middle_squares, c=color)
plt.show()

for step, color in steps_and_colors:
    trapezoid_squares = []
    s_trap = trapezoid_integration(x_min, x_max, step)
    plt.plot(range(len(trapezoid_squares)), trapezoid_squares, c=color)
plt.show()

for step, color in steps_and_colors:
    simpson_squares = []
    s_sim = simpson_integration(x_min, x_max, step)
    plt.plot(range(len(simpson_squares)), simpson_squares, c=color)
plt.show()
