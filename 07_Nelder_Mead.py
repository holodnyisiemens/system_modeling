'''Метод Нелдера-Мида, также известный как метод деформируемого 
многогранника и симплекс-метод, — метод безусловной оптимизации 
функции от нескольких переменных, не использующий производной (точнее 
— градиентов) функции, а поэтому легко применим к негладким и/или 
зашумлённым функциям. Суть метода заключается в последовательном 
перемещении и деформировании симплекса вокруг точки экстремума. Метод 
находит локальный экстремум и может «застрять» в одном из них. Если всё же 
требуется найти глобальный экстремум, можно пробовать выбирать другой 
начальный симплекс.'''

import random
import pandas as pd
import numpy as np

# исходная функция, которую требуется оптимизировать
def func(x, z):
    # отброс периодической части
    return np.sin(z % (2 * np.pi)) + np.exp(-x)

# FIXME заменить на двумерный массив
# симплекс - фигура вокруг точки, при помощи которой будет оптимизироваться функция
class Simplex:
    def __init__(self, p1, p2, p3, p4) -> None:
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    # кортеж значений функции в точках
    def get_func_vals(self) -> tuple:
        return (
            func(self.p1[0], self.p1[2]),
            func(self.p2[0], self.p2[2]), 
            func(self.p3[0], self.p3[2]),
            func(self.p4[0], self.p4[2])
        )

    # список кортежей из точек и значений фукнции y(x, z) в них
    def get_points_and_func_vals(self) -> list:
        return list(zip((self.p1, self.p2, self.p3, self.p4), self.get_func_vals()))

    def get_variance(self) -> float:
        # вычисление средней точки симплекса
        avg_point = np.empty(shape=(3, ))
        avg_point = (self.p1 + self.p2 + self.p3 + self.p4) / 4

        # FIXME
        # сумма квадратов расстояний от каждой точки симплекса до центральной точки
        sum_of_squares = (
            (self.p1[0] - avg_point[0]) ** 2 + (self.p1[1] - avg_point[1]) ** 2 + (self.p1[2] - avg_point[2]) ** 2 +
            (self.p2[0] - avg_point[0]) ** 2 + (self.p2[1] - avg_point[1]) ** 2 + (self.p2[2] - avg_point[2]) ** 2 +
            (self.p3[0] - avg_point[0]) ** 2 + (self.p3[1] - avg_point[1]) ** 2 + (self.p3[2] - avg_point[2]) ** 2 +
            (self.p4[0] - avg_point[0]) ** 2 + (self.p4[1] - avg_point[1]) ** 2 + (self.p4[2] - avg_point[2]) ** 2
        )

        # дисперсия
        my_variance = sum_of_squares / 4
        # print('my:', my_variance)


        points = np.array((self.p1, self.p2, self.p3, self.p4))
        # print(points)
        variance_x = np.var(points[:, 0])
        variance_y = np.var(points[:, 1])
        variance_z = np.var(points[:, 2])
        total_variance = np.var(points)

        variance = total_variance
        # print('np:', variance)

        return my_variance

'''ШАГ 1: Вначале выбирается n+1 точка 𝑥 = (𝑥(1), 𝑥(2), … , 𝑥(𝑛)), 
𝑖 = 1 … 𝑛 + 1, образующие симплекс n-мерного пространства. 
В этих точках вычисляются значения функции: 𝑓1 = 𝑓(𝑥1), 𝑓2 = 𝑓(𝑥2), … , 𝑓𝑛 = 𝑓(𝑥𝑛)'''
def preparing(x_start: float, z_start: float, simplex_step: float) -> Simplex:
    y_start = func(x_start, z_start)
    # т.к. точки представлены в 3-мерном пространстве, необходимо 4 точки для симплекса (тетраедр)
    # точки симплекса откладываются с шагом simplex_step отностительно исходной точки
    center = np.array((x_start, y_start, z_start))
    p1 = np.array((center[0], center[1], center[2] + simplex_step))
    p2 = np.array((center[0], center[1] + simplex_step, center[2] - simplex_step))
    p3 = np.array((center[0] - simplex_step, center[1] - simplex_step, center[2] - simplex_step))
    p4 = np.array((center[0] + simplex_step, center[1] - simplex_step, center[2] - simplex_step))

    return Simplex(p1, p2, p3, p4)

'''ШАГ 2: Из вершин симплекса выбираются: точка с наибольшим (из выбранных) значением
функции, точка со следующим по величине значением функции и точка с наименьшим значением функции. 
Целью дальнейших манипуляций будет уменьшение по крайней мере наибольшего значения фукнции.'''
def sorting(points_and_func_vals: list) -> list:
    # сортировка кортежей точек и значений функции по возрастанию значений функции
    # 3-й индекс: точка с наибольшим значением функции
    # 2-й индекс: точка со следующим по величине значением
    # 0-й индекс: точка с наименьшим значением функции
    # возвращается список кортежей искомых точек и соответствующих им значений
    sorted_points_and_func_vals = sorted(points_and_func_vals, key=lambda x: x[1])
    return sorted_points_and_func_vals

'''ШАГ 3: Определения центра тяжести всех точек, кроме точки, с максимальным значением функции
Центр тяжести находится в точке, координаты которой равны средним значениям сумм координат'''
def get_gravity_center(sorted_points_and_func_vals):
    # гравитационный центр
    xc = np.array((0., 0., 0.))
    # все точки кроме последней (с максимальным значением функции)
    for p, _ in sorted_points_and_func_vals[:3]:
        xc += 1/3 * p
    return xc

'''ШАГ 4: Отражение. Отразим точку с наибольшим значением функции относительно центра тяжести
с коэффициентом 𝛼 (при 𝛼 = 1 это будет центральная симметрия, в общем случае — гомотетия), 
и в полученной точке вычислим функцию'''
def reflection(xc, xh, a=1) -> tuple:
    # отраженная точка
    xr = np.empty(shape=xc.shape)
    xr = (1 + a) * xc - a * xh

    # кортеж из искомой точки и значения функции в ней
    return xr, func(xr[0], xr[2])

'''ШАГ 6: Сжатие'''
def compression(xc, xh, b=0.5) -> tuple:
    xs = np.empty(shape=xc.shape)
    xs = (1 - b) * xc + b * xh
    return xs, func(xs[0], xs[2])

'''ШАГ 5: Определение места fr в ряду fh, fg и fl'''
def define_refl_val_place(xc, xr, fr: float, xh, fh: float, fg: float, fl: float, gamma: float = 2) -> tuple:
    # флаг продолжения итерации при оптимизации
    continue_iter = True
    if fr <= fl:
        # направление выбрано удачное и можно попробовать увеличить шаг (растяжение)
        xe = np.empty(shape=xc.shape)
        xe = (1 - gamma) * xc + gamma * xr
        fe = func(xe[0], xe[2])

        if fe <= fr:
            # можно расширить симплекс до этой точки
            xh = xe
        elif fr < fe:
            # переместились слишком далеко
            xh = xr

        # завершение итерации (переход на 9 шаг)
        continue_iter = False
    elif fl < fr < fg:
        # новая точка лучше двух прежних
        xh = xr
        # завершение итерации (переход на 9 шаг)
        continue_iter = False
    elif fg <= fr < fh:
        # меняем местами значения 𝑥𝑟 с 𝑥ℎ, 𝑓𝑟 с 𝑓ℎ
        # верхний предел не включен, т.к. если значения будут равны, менять их местами не будет смысла 
        xr, xh = xh, xr
    
    # в остальных случаях fh <= fr

    fr = func(xr[0], xr[2])
    fh = func(xh[0], xh[2])
    return xr, fr, xh, fh, continue_iter

# оптимизация функции
def nelder_mead_optimize(x_start: float = None, z_start: float = None, simplex_step: float = None, sorted_p_and_f: list = None, acc: float = None) -> tuple[float, float]:
    try:
        # каждая итерация кроме первой начинается с шага 2
        if not sorted_p_and_f:
            # ШАГ 1: Подготовка
            simplex = preparing(x_start, z_start, simplex_step)
            # ШАГ 2: Сортировка
            sorted_p_and_f = sorting(simplex.get_points_and_func_vals())
        else:
            # точка симплекса с наибольшим значением функции и само значение функции в ней
            # ШАГ 2: Сортировка
            sorted_p_and_f = sorting(sorted_p_and_f)

        # точка симплекса с наибольшим значением функции и само значение функции в ней
        xh, fh = sorted_p_and_f[3]
        # со вторым по величине
        xg, fg = sorted_p_and_f[2]
        # с наименьшим
        xl, fl = sorted_p_and_f[0]

        # ШАГ 3: Получение гравитационного гравитационного центра
        xc = get_gravity_center(sorted_p_and_f)

        # ШАГ 4: Отражение
        xr, fr = reflection(xc, xh, a=1)

        # ШАГ 5
        xr, fr, xh, fh, continue_iter = define_refl_val_place(xc, xr, fr, xh, fh, fg, fl)

        # обновление значений
        sorted_p_and_f[3] = xh, fh
        sorted_p_and_f[2] = xg, fg
        sorted_p_and_f[0] = xl, fl

        # проверка необходимости выполнения 7 и 8 шага
        if continue_iter:
            # ШАГ 6: Сжатие
            xs, fs = compression(xc, xh)

            # ШАГ 7
            if fs <= fh:
                xh = xs
                fh = fs
                sorted_p_and_f[3] = xh, fh

            # ШАГ 8
            else:
                # fs > fh, первоначальные точки оказались самыми удачными
                # делаем «глобальное сжатие» симплекса — гомотетию к точке с наименьшим значением
                for i in range(1, len(sorted_p_and_f)):
                    p, _ = sorted_p_and_f[i]
                    p = (p + xl) / 2
                    sorted_p_and_f[i] = (p, func(p[0], p[2]))

        # переопределение симлекса (извлекаются только точки из нового списка точек и значений функции)
        p1, p2, p3, p4 = (p[0] for p in sorted_p_and_f[:4])
        simplex = Simplex(p1, p2, p3, p4)

        # ШАГ 9: условие сходимости (проверка взаимной близости полученных вершин симплекса через дисперсию, что предполагает и близость их к искомому минимуму)
        if acc and (simplex.get_variance() < acc):
            return sorted_p_and_f[0][0][0], sorted_p_and_f[0][0][2]

        return nelder_mead_optimize(sorted_p_and_f=sorted_p_and_f, acc=acc)

    except RecursionError:
        # при возникновении ошибки рекурсии возвращаются текущие оптимальные значения
        return sorted_p_and_f[0][0][0], sorted_p_and_f[0][0][2]

# 30 тестов 
x_data = np.random.randint(-1000, 1000, size=30)
z_data = np.random.randint(-1000, 1000, size=30)

# точность и шаг отступа от точки для построения симплекса вокруг нее
acc = 0.00000001
simplex_step = 0.01

# таблица с результатами
data = []

for x_start, z_start in zip(x_data, z_data):
    # начальное значение функции
    y_start = func(x_start, z_start)

    # значения после оптимизации
    opt_x, opt_z = nelder_mead_optimize(x_start=x_start, z_start=z_start, simplex_step=simplex_step, acc=acc)
    opt_y = func(opt_x, opt_z)

    # проверка на уменьшение найденного минимума функции и соответствие с ожидаемым результатом (-1)
    if opt_y < -0.9 and opt_y < y_start:
        test_res = 'PASS'
    else:
        test_res = 'FAIL'

    # строка таблицы (для читаемости  оптимизированный y округляется до 4х знаков после запятой)
    data.append((x_start, z_start, y_start, opt_x, opt_z, round(opt_y, 4), test_res))

pd.options.display.float_format = lambda num: 'inf' if num > 1e6 else '{:.3f}'.format(num)
table = pd.DataFrame(data, columns=['x_start', 'z_start', 'y_start', 'opt_x', 'opt_z', 'opt_y', 'test_res'])
print(table)

# количество успешно пройденных тестов
print('PASSED:', len(table[table['test_res']=='PASS']))
