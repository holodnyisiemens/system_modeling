from math import sin, e, pi, sqrt, cos, isnan
import random
import pandas as pd
from dataclasses import dataclass
import time

# экспонента
def exp(num: float) -> float:
    # при переполнении возвращается бесконечность
    try:
        return e ** num
    except OverflowError:
        return float('inf')

# исходная функция, которую требуется оптимизировать
def func(x: float, z: float) -> float:
    # отброс периодической части
    return sin(z % (2*pi)) + exp(-x)

'''Многомерный метод Ньютона'''

# первая частная производная y по x
def get_first_part_deriv_y_on_x(x: float) -> float:
    return -exp(-x)

# первая частная производная y по z
def get_first_part_deriv_y_on_z(z: float) -> float:
    # отброс периодической части
    return cos(z % (2*pi))

# вычисление Гессиана
def get_hessian(x: float, z: float) -> float:
    # отброс периодической части
    return -sin(z % (2*pi)) * exp(-x)

# получение градиента (вектора частных производных)
def get_gradient(x: float, z: float) -> tuple:
    return get_first_part_deriv_y_on_x(x), get_first_part_deriv_y_on_z(z)

# оптимизация функции sin(z) + e^-x (поиск точки минимума)
def newton_optimize(x: float, z: float, acc: float = None) -> tuple:
    x_grad, z_grad = get_gradient(x, z)
    try:
        # проверка нормы градиента
        if acc and (sqrt(x_grad ** 2 + z_grad ** 2) < acc):
            return x, z
        # при возникновении ошибок вычисления нормы градиента
    
        # гессиан
        hess = get_hessian(x, z)

        new_x = x - x_grad / hess
        new_z = z - z_grad / hess
        
        # проверка на значения NaN (Not a Number - не число)
        if isnan(new_x) and isnan(new_z):
            return x, z
        elif isnan(new_x):
            return newton_optimize(x, new_z, acc)
        elif isnan(new_z):
            return newton_optimize(new_x, z, acc)
        else:
            return newton_optimize(new_x, new_z, acc)
    except:
        # при возникновении какой-либо ошибки (влючая ошибку рекурсии и деления на ноль), возвращаются текущие оптимальные значения: 
        return x, z

'''Метод градиентного спуска'''

# норма градиента
def get_gradient_norm(x_grad, z_grad):
    return sqrt(x_grad ** 2 + z_grad ** 2)

# функция зависимости от шага, являющаяся частью градиентного спуска, которую нужно минимизировать (для поиска локального минимума)
def func_need_argmin_descent_speed(x, z, x_descent_speed, z_descent_speed, x_grad, z_grad):
    return (
        func(x - x_descent_speed * x_grad, z),
        func(x, z - z_descent_speed * z_grad)
    )

# тейлоровская аппроксимация первого порядка с гиперпараметрами a и b, регулирующими наклоны функций ф1 и ф2
# используется норма градиента
def first_order_taylor_approx_with_hyperparam(hyperparam, x, z, x_descent_speed, z_descent_speed, x_grad, z_grad):
    return (
        func(x, z) - hyperparam * x_descent_speed * get_gradient_norm(x_grad, z_grad) ** 2,
        func(x, z) - hyperparam * z_descent_speed * get_gradient_norm(x_grad, z_grad) ** 2
    )

# проверяем условия неточной оптимизации: ф1(шаг) <= ф(шаг) <= ф2(шаг), p берут в диапазоне (0.5, 1)
def check_conditions_inaccurate_optimization(x, z, x_descent_speed, z_descent_speed, x_grad, z_grad, p=0.6):
    a = p
    b = 1 - p
    fi1x, fi1z = first_order_taylor_approx_with_hyperparam(a, x, z, x_descent_speed, z_descent_speed, x_grad, z_grad)
    fix, fiz = func_need_argmin_descent_speed(x, z, x_descent_speed, z_descent_speed, x_grad, z_grad)
    fi2x, fi2z = first_order_taylor_approx_with_hyperparam(b, x, z, x_descent_speed, z_descent_speed, x_grad, z_grad)
    return (fi1x <= fix and fix <= fi2x), (fi1z <= fiz and fiz <= fi2z)

# gamma - понижающий коэф в диапазоне (0, 1)
def get_fastest_descent_speed(x, z, x_descent_speed, z_descent_speed, x_grad, z_grad, gamma=0.7):
    # поочередное изменение шага по x и по z
    while not check_conditions_inaccurate_optimization(x, z, x_descent_speed, z_descent_speed, x_grad, z_grad)[0]:
        x_descent_speed *= gamma
    while not check_conditions_inaccurate_optimization(x, z, x_descent_speed, z_descent_speed, x_grad, z_grad)[1]:
        z_descent_speed *= gamma

    return x_descent_speed, z_descent_speed

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

# оптимизация функции sin(z) + e^-x (поиск точки минимума)
def grad_descent_optimize(x, z, x_descent_speed, z_descent_speed, acc=None):
    x_grad, z_grad = get_gradient(x, z)
    try:
        # проверка нормы градиента
        if acc and (get_gradient_norm(x_grad, z_grad) < acc):
            return x, z
    except:
        # при возникновении ошибок вычисления нормы градиента
        return x, z

    # рекурсия (если не указана точность, то будет выполняться до возникновения ошибки рекурсии)
    try:
        new_x = x - x_grad / x_descent_speed
        new_z = z - z_grad / z_descent_speed
        x_descent_speed, z_descent_speed = get_fastest_descent_speed(new_x, new_z, x_descent_speed, z_descent_speed, x_grad, z_grad)
        return grad_descent_optimize(new_x, new_z, x_descent_speed, z_descent_speed, acc)
    except RecursionError:
        return x, z

'''Метод сопряженных градиентов'''

# поправочный коэффициент w
def get_correction_coeff(x: float, z: float, x_old: float, z_old: float) -> tuple:
    x_grad, z_grad = get_gradient(x, z)
    x_grad_old, z_grad_old = get_gradient(x_old, z_old)
    norm_x_grad, norm_z_grad = get_gradient_norm(x_grad, z_grad)
    norm_x_grad_old, norm_z_grad_old = get_gradient_norm(x_grad_old, z_grad_old)
    w_x = (norm_x_grad / norm_x_grad_old) ** 2
    w_z = (norm_z_grad / norm_z_grad_old) ** 2
    return w_x, w_z

# сопряженный вектор S
def get_s(x: float, z: float, x_old: float, z_old: float, s_old_x: float, s_old_z: float) -> tuple:
    # поправочные коэффициенты w
    w_x, w_z = get_correction_coeff(x, z, x_old, z_old)
    new_s_x = - get_gradient(x, z) + w_x * s_old_x
    new_s_z = - get_gradient(x, z) + w_z * s_old_z
    return new_s_x, new_s_z

def conjugate_grad_optimize(x: float, z: float, lamb: float, s_old_x: float = None, s_old_z: float = None, acc: float = None) -> tuple:    
    try:
        x_grad, z_grad = get_gradient(x, z)

        # условие сходимости
        if acc and (get_gradient_norm(x_grad, z_grad) < acc):
            return x, z
        
        # получение начального сопряженного вектора на первой итерации
        if not (s_old_x and s_old_z):
            s_old_x = - x_grad
            s_old_z = - z_grad

        x += lamb * s_old_x
        z += lamb * s_old_z

        return conjugate_grad_optimize(x, z, lamb, s_old_x, s_old_z)
    except RecursionError:
        # при возникновении ошибки рекурсии возвращаются текущие оптимальные значения
        return x, z

'''Метод Нелдера-Мида'''

# класс для произвольной точки
@dataclass
class Point:
    x: float
    y: float
    z: float

# симплекс - фигура вокруг точки, при помощи которой будет оптимизироваться функция
@dataclass
class Simplex:
    p1: Point
    p2: Point
    p3: Point
    p4: Point

    # кортеж значений функции в точках
    def get_func_vals(self) -> tuple:
        return (
            func(self.p1.x, self.p1.z),
            func(self.p2.x, self.p2.z), 
            func(self.p3.x, self.p3.z),
            func(self.p4.x, self.p4.z)
        )

    # список кортежей из точек и значений фукнции y(x, z) в них
    def get_points_and_func_vals(self) -> list:
        return list(zip((self.p1, self.p2, self.p3, self.p4), self.get_func_vals()))

    def get_variance(self) -> float:
        # вычисление средней точки симплекса
        avg_point = Point(
            (self.p1.x + self.p2.x + self.p3.x + self.p4.x) / 4,
            (self.p1.y + self.p2.y + self.p3.y + self.p4.y) / 4,
            (self.p1.z + self.p2.z + self.p3.z + self.p4.z) / 4
        )

        # сумма квадратов расстояний от каждой точки симплекса до центральной точки
        sum_of_squares = (
            (self.p1.x - avg_point.x) ** 2 + (self.p1.y - avg_point.y) ** 2 + (self.p1.z - avg_point.z) ** 2 +
            (self.p2.x - avg_point.x) ** 2 + (self.p2.y - avg_point.y) ** 2 + (self.p2.z - avg_point.z) ** 2 +
            (self.p3.x - avg_point.x) ** 2 + (self.p3.y - avg_point.y) ** 2 + (self.p3.z - avg_point.z) ** 2 +
            (self.p4.x - avg_point.x) ** 2 + (self.p4.y - avg_point.y) ** 2 + (self.p4.z - avg_point.z) ** 2
        )

        # дисперсия
        variance = sum_of_squares / 4

        return variance

'''ШАГ 1: Вначале выбирается n+1 точка 𝑥 = (𝑥(1), 𝑥(2), … , 𝑥(𝑛)), 
𝑖 = 1 … 𝑛 + 1, образующие симплекс n-мерного пространства. 
В этих точках вычисляются значения функции: 𝑓1 = 𝑓(𝑥1), 𝑓2 = 𝑓(𝑥2), … , 𝑓𝑛 = 𝑓(𝑥𝑛)'''
def preparing(x_start: float, z_start: float, simplex_step: float) -> Simplex:
    y_start = func(x_start, z_start)
    # т.к. точки представлены в 3-мерном пространстве, необходимо 4 точки для симплекса (тетраедр)
    # точки симплекса откладываются с шагом simplex_step отностительно исходной точки
    center = Point(x_start, y_start, z_start)
    p1 = Point(center.x, center.y, center.z + simplex_step)
    p2 = Point(center.x, center.y + simplex_step, center.z - simplex_step)
    p3 = Point(center.x - simplex_step, center.y - simplex_step, center.z - simplex_step)
    p4 = Point(center.x + simplex_step, center.y - simplex_step, center.z - simplex_step)

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
def get_gravity_center(sorted_points_and_func_vals: list) -> Point:
    # гравитационный центр
    xc = Point(0, 0, 0)
    # все точки кроме последней (с максимальным значением функции)
    for p, _ in sorted_points_and_func_vals[:3]:
        xc.x += 1/3 * p.x
        xc.y += 1/3 * p.y
        xc.z += 1/3 * p.z
    return xc

'''ШАГ 4: Отражение. Отразим точку с наибольшим значением функции относительно центра тяжести
с коэффициентом 𝛼 (при 𝛼 = 1 это будет центральная симметрия, в общем случае — гомотетия), 
и в полученной точке вычислим функцию'''
def reflection(xc: Point, xh: Point, a=1) -> tuple:
    # отраженная точка
    xr = Point(
        (1 + a) * xc.x - a * xh.x,
        (1 + a) * xc.y - a * xh.y,
        (1 + a) * xc.z - a * xh.z
    )

    # кортеж из искомой точки и значения функции в ней
    return xr, func(xr.x, xr.z)

'''ШАГ 6: Сжатие'''
def compression(xc: Point, xh: Point, b=0.5) -> tuple:
    xs = Point(
        (1 - b) * xc.x + b * xh.x,
        (1 - b) * xc.y + b * xh.y,
        (1 - b) * xc.z + b * xh.z
    )
    return xs, func(xs.x, xs.z)

'''ШАГ 5: Определение места fr в ряду fh, fg и fl'''
def define_refl_val_place(xc: Point, xr: Point, fr: float, xh: Point, fh: float, fg: float, fl: float, gamma: float = 2) -> tuple:
    # флаг продолжения итерации при оптимизации
    continue_iter = True
    if fr <= fl:
        # направление выбрано удачное и можно попробовать увеличить шаг (растяжение)
        xe = Point(
            (1 - gamma) * xc.x + gamma * xr.x,
            (1 - gamma) * xc.y + gamma * xr.y,
            (1 - gamma) * xc.z + gamma * xr.z
        )
        fe = func(xe.x, xe.z)

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

    fr = func(xr.x, xr.z)
    fh = func(xh.x, xh.z)
    return xr, fr, xh, fh, continue_iter

# оптимизация функции
def nelder_mead_optimize(x_start: float = None, z_start: float = None, simplex_step: float = None, sorted_p_and_f: list = None, acc: float = None) -> tuple:
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
                    p.x = (p.x + xl.x) / 2
                    p.y = (p.y + xl.y) / 2
                    p.z = (p.z + xl.z) / 2
                    sorted_p_and_f[i] = (p, func(p.x, p.z))

        # переопределение симлекса (извлекаются только точки из нового списка точек и значений функции)
        p1, p2, p3, p4 = (p[0] for p in sorted_p_and_f[:4])
        simplex = Simplex(p1, p2, p3, p4)

        # ШАГ 9: условие сходимости (проверка взаимной близости полученных вершин симплекса через дисперсию, что предполагает и близость их к искомому минимуму)
        if acc and (simplex.get_variance() < acc):
            return sorted_p_and_f[0][0].x, sorted_p_and_f[0][0].z

        return nelder_mead_optimize(sorted_p_and_f=sorted_p_and_f, acc=acc)

    except RecursionError:
        # при возникновении ошибки рекурсии возвращаются текущие оптимальные значения
        return sorted_p_and_f[0][0].x, sorted_p_and_f[0][0].z

# 30 тестов
# случайные начальные значения x и z
x_data = [random.randint(-100, 100) for _ in range(31)]
z_data = [random.randint(-100, 100) for _ in range(31)]

# шаг сходимости
lamb = 0.01
# точность
acc = 0.000001
# шаг формирования симплекса относительно исходной точки с ранее описанными случайными значениями x, z, а также вычисленным по ним y
simplex_step = 0.01

# таблица с результатами
data = []

for x_start, z_start in zip(x_data, z_data):
    y_start = func(x_start, z_start)

    start_time = time.perf_counter_ns()
    newton_opt_x, newton_opt_z = newton_optimize(x=x_start, z=z_start, acc=acc)
    end_time = time.perf_counter_ns()
    newton_time = (end_time - start_time) / 1e6
    newton_opt_y = func(newton_opt_x, newton_opt_z)

    start_time = time.perf_counter_ns()
    grad_desc_opt_x, grad_desc_opt_z = grad_descent_optimize(x=x_start, z=z_start, x_descent_speed=lamb, z_descent_speed=lamb, acc=acc)
    end_time = time.perf_counter_ns()
    grad_desc_time = (end_time - start_time) / 1e6
    grad_desc_opt_y = func(grad_desc_opt_x, grad_desc_opt_z)

    start_time = time.perf_counter_ns()
    conj_grad_opt_x, conj_grad_opt_z = conjugate_grad_optimize(x=x_start, z=z_start, lamb=lamb, acc=acc)
    end_time = time.perf_counter_ns()
    conj_grad_time = (end_time - start_time) / 1e6
    conj_grad_opt_y = func(conj_grad_opt_x, conj_grad_opt_z)

    start_time = time.perf_counter_ns()
    nelder_mead_opt_x, nelder_mead_opt_z = nelder_mead_optimize(x_start=x_start, z_start=z_start, simplex_step=simplex_step, acc=acc)
    end_time = time.perf_counter_ns()
    nelder_mead_time = (end_time - start_time) / 1e6
    nelder_mead_opt_y = func(nelder_mead_opt_x, nelder_mead_opt_z)

    min_opt_y = min(newton_opt_y, grad_desc_opt_y, conj_grad_opt_y, nelder_mead_opt_y)
    if min_opt_y == newton_opt_y:
        best_value = 'newton'
    elif min_opt_y == grad_desc_opt_y:
        best_value = 'grad descent'
    elif min_opt_y == conj_grad_opt_y:
        best_value = 'conj grads'
    elif min_opt_y == nelder_mead_opt_y:
        best_value = 'Nelder-Mead'

    min_time = min(newton_time, grad_desc_time, conj_grad_time, nelder_mead_time)
    if min_time == newton_time:
        best_time = 'newton'
    elif min_time == grad_desc_time:
        best_time = 'grad descent'
    elif min_time == conj_grad_time:
        best_time = 'conj grads'
    elif min_time == nelder_mead_time:
        best_time = 'Nelder-Mead'

    # добавление очередной строки
    data.append((x_start, z_start, y_start, newton_opt_y, f'{newton_time}ms', grad_desc_opt_y, f'{grad_desc_time}ms', conj_grad_opt_y, f'{conj_grad_time}ms', nelder_mead_opt_y, f'{nelder_mead_time}ms', best_value, best_time))

pd.options.display.float_format = lambda num: 'inf' if num > 1e6 else '{:.3f}'.format(num)
table = pd.DataFrame(data, columns=['x_start', 'z_start', 'y_start', 'newton_y', 'newton_t', 'grad_desc_y', 'grad_desc_t', 'conj_grad_y', 'conj_grad_t', 'nelder_mead_y', 'nelder_mead_t', 'best_value', 'best_time'])
print(table)
print('Best "newton" value count = ', len(table[(table['best_value']=='newton')]))
print('Best "grad descent" value count = ', len(table[(table['best_value']=='grad descent')]))
print('Best "conj grads" value count = ', len(table[(table['best_value']=='conj grads')]))
print('Best "Nelder-Mead" value count = ', len(table[(table['best_value']=='Nelder-Mead')]))

print('Best "newton" time count = ', len(table[(table['best_time']=='newton')]))
print('Best "grad descent" time count = ', len(table[(table['best_time']=='grad descent')]))
print('Best "conj grads" time count = ', len(table[(table['best_time']=='conj grads')]))
print('Best "Nelder-Mead" time count = ', len(table[(table['best_time']=='Nelder-Mead')]))
