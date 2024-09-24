'''Геометрический алгоритм Монте-Карло для численного интегрирования'''

from math import atan
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# функция, которую нужно интегрировать
def f(x: float) -> float:
    return (x ** 5 - x + 1) / (x ** 2 + 1)

# первообразная функции
def antiderivative(x: float) -> float:
    return atan(x) + x ** 4 / 4 - x ** 2 / 2

# аналитическое значение интеграла исходной функции
def analytical_integration(x_min: float, x_max: float) -> float:
    return antiderivative(x_max) - antiderivative(x_min)

# поиск максимума функции на заданном диапазоне
def get_func_max_in_range(x_min: float, x_max: float, step: float) -> float:
    func_max = None
    for x in np.arange(x_min, x_max + step, step):
        value = f(x)
        if not func_max:
            func_max = value
        elif value > func_max:
            func_max = value
    return func_max

# метод Монте-Карло
def monte_carlo_integration(x_min: float, x_max: float, step: float, m: int, N: int) -> float:
    # длина прямоугольника, в который будут запускаться точки
    L = x_max - x_min

    # ширина прямоугольника
    H = get_func_max_in_range(x_min, x_max, step)

    # площадь прямоугольника    
    S_rect = H * L

    # площадь под графиком - интеграл
    S = 0

    # m - число экспериментов
    for _ in range(m):
        # количество точек, попавших под график функции
        K = 0

        # N - число случайных точек, распространяемых по прямоугольнику
        for _ in range(N):
            # случайные значения от 0 до 1
            gamma1 = random.random()
            gamma2 = random.random()

            # генерируемая точка
            Xi = Point(x_min + L * gamma1, H * gamma2)

            # проверка на вхождение под график
            if Xi.y <= f(Xi.x):
                K += 1

        # доля точек, входящих под график от числа всех точек
        pi = K / N

        # значение площади при текущем эксперименте
        Si = S_rect * pi
        
        # среднее значение площади
        S += 1 / m * Si
    return S

x = np.arange(-20, 20)
y = [f(xi) for xi in x]

plt.plot(x, y)
plt.title('Исходная функция')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

# тесты для разного числа экспериментов и количества точек
x_min = 0
x_max = 10
step = 0.01
experiments_nums = [1, 10, 100, 1000]
point_nums = [10, 100, 1000, 10000]
points_and_experiments_nums = [(e_num, p_num) for p_num in point_nums for e_num in experiments_nums]

# таблица с результатами
data = []

for e_num, p_num in points_and_experiments_nums:
    # значения интеграла
    s_analyt = analytical_integration(x_min, x_max)
    s_monte_carlo = monte_carlo_integration(x_min, x_max, step, e_num, p_num)
    data.append((x_min, x_max, step, e_num, p_num, s_analyt, s_monte_carlo))

table = pd.DataFrame(data, columns=['x_min', 'x_max', 'step', 'e_num', 'p_num', 's_analyt', 's_monte_carlo'])

# если отностильная погрешность меньше процента, интеграл считается верно рассчитанным
accuracy = 0.01
table['error'] = abs(table['s_monte_carlo'] - table['s_analyt']) / abs(table['s_monte_carlo'])
table['result'] = np.where(table['error'] < accuracy, 'PASS', 'FAIL')

print(table)

# число успешно пройденных тестов
print('PASSED:', (table['error'] == 'PASSED').sum())
