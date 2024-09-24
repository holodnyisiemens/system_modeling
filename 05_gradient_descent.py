'''Задачи:
1. Реализовать градиентный спуск для функции y = sin(z) + e^-x при постоянной и уменьшающейся скорости спуска.
2. Предложить алгоритм подбора шагов для наискорейшего спуска.
3. Сопоставить результаты с работой готовых функций градиентного спуска из библиотек Python.'''

# см. формулы для наискорейшего градиентного спуска
# наискорейший град спуск соблюдается, если шаг - это минимум функции в след точке (которую также можно выразить через градиентный спуск)
# минимизировать эту функцию можно следующим образом:
 
# см. правило Армихо-Гольдштейна и алгоритм Backtracking
# тейлоровская аппроксимация первого порядка с гиперпараметрами a и b, регулирующими наклоны функций ф1 и ф2
# обычно a берется как некоторое p в пределах (0.5, 1) и b = 1 - p
# итоговое значение функции должно быть в рамках этих аппроксимаций
# сначала задаем начальный шаг
# проверяем условия неточной оптимизации: ф1(шаг) <= ф(шаг) <= ф2(шаг)
# где ф - функция зависимости от шага, являющаяся частью градиентного спуска
# если выполнено то оставляем этот шаг (не обязательно оптимальный, но достаточно быстро находимый)
# если нет, то умножаем на некоторый понижающий коэф (делаем Backtracking)
# продолжаем пока не найдем подходящий шаг

from math import sin, cos, e, sqrt, pi
import pandas as pd
import random
from scipy.optimize import minimize
import tensorflow as tf

# отключение сообщений об ошибках при работе TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# норма градиента
def get_gradient_norm(x_grad, z_grad):
    return sqrt(x_grad ** 2 + z_grad ** 2)

# функция зависимости от шага, являющаяся частью градиентного спуска, которую нужно минимизировать (для поиска локального минимума)
def func_need_argmin_descent_speed(x, z, x_descent_speed, z_descent_speed, x_grad, z_grad):
    return (
        f(x - x_descent_speed * x_grad, z),
        f(x, z - z_descent_speed * z_grad)
    )

# тейлоровская аппроксимация первого порядка с гиперпараметрами a и b, регулирующими наклоны функций ф1 и ф2
# используется норма градиента
def first_order_taylor_approx_with_hyperparam(hyperparam, x, z, x_descent_speed, z_descent_speed, x_grad, z_grad):
    return (
        f(x, z) - hyperparam * x_descent_speed * get_gradient_norm(x_grad, z_grad) ** 2,
        f(x, z) - hyperparam * z_descent_speed * get_gradient_norm(x_grad, z_grad) ** 2
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
def optimize(x, z, x_descent_speed, z_descent_speed, descent_type, acc=None):
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
        if descent_type == 'decreasing':
            # уменьшение шага при каждом приближении к оптимальному решению
            x_descent_speed *= 0.8
            z_descent_speed *= 0.8
        elif descent_type == 'fastest':
            x_descent_speed, z_descent_speed = get_fastest_descent_speed(new_x, new_z, x_descent_speed, z_descent_speed, x_grad, z_grad)
        elif descent_type == 'const':
            pass
        else:
            raise ValueError('Incorrect value for speed type')
        return optimize(new_x, new_z, x_descent_speed, z_descent_speed, descent_type, acc)
    except RecursionError:
        return x, z

# вспомогательная функция для передачи в функцию scipy minimize (требуется с одним параметром)
def pre_function(params):
    x, z = params
    return f(x, z)

# минимизация с помощью scipy
def scipy_minimize(initial_guess):
    result = minimize(pre_function, initial_guess, method='L-BFGS-B')
    return result.x[0], result.x[1], result.fun

# точность
acc=0.00001

# шаг сходимости
x_descent_speed = z_descent_speed = learning_rate = 0.01

# перевод функции в TensorFlow (иначе не будет считаться градиент)
def tf_f(x, z):
    return tf.sin(z % (2.0 * pi)) + tf.exp(-x)

# оптимизация с помощью Tensorflow
def tensorflow_opt(learning_rate):
    for _ in range(1000):
        # Вычисление градиентов
        with tf.GradientTape() as tape:
            loss = tf_f(x, z)
        gradients = tape.gradient(loss, [x, z])

        # Обновление переменных с учетом градиентов
        x.assign_sub(learning_rate * gradients[0])
        z.assign_sub(learning_rate * gradients[1])
    # Получение оптимальных значений
    opt_x = x.numpy()
    opt_z = z.numpy()
    opt_y = f(opt_x, opt_z)
    return opt_x, opt_z, opt_y

# тесты
z_data_1 = [random.uniform(3/2*pi-0.8, 3/2*pi+0.8) for _ in range(16)]
z_data_2 = [random.uniform(-pi/2-0.8, -pi/2+0.8) for _ in range(16)]
z_data = z_data_1 + z_data_2
x_data = [random.randint(0, 700) for _ in range(31)]

# таблица с результатами
data = []

for x_start, z_start in zip(x_data, z_data):
    my_conts_opt_x, my_const_opt_z = optimize(x_start, z_start, x_descent_speed, z_descent_speed, descent_type='const', acc=acc)
    my_decr_opt_x, my_decr_opt_z = optimize(x_start, z_start, x_descent_speed, z_descent_speed, descent_type='decreasing', acc=acc)
    my_fast_opt_x, my_fast_opt_z = optimize(x_start, z_start, x_descent_speed, z_descent_speed, descent_type='fastest', acc=acc)

    my_const_opt_y = f(my_conts_opt_x, my_const_opt_z)
    my_decr_opt_y = f(my_decr_opt_x, my_decr_opt_z)
    my_fast_opt_y = f(my_fast_opt_x, my_fast_opt_z)

    # начальные переменные для Tensorflow (градиент будет рассчитываться только для этого типа)
    x = tf.Variable(float(x_start), dtype=tf.float64)
    z = tf.Variable(float(z_start), dtype=tf.float64)
    tf_opt_x, tf_opt_z, tf_opt_y = tensorflow_opt(learning_rate=learning_rate)

    # изначальные значения x и z для оптимизации через scipy
    initial_guess=[x_start, z_start]
    scipy_opt_x, scipy_opt_z, scipy_opt_y = scipy_minimize(initial_guess)
    
    # при очень больших значениях y - принимается бесконечностью
    if my_const_opt_y and my_const_opt_y > 1000000000:
        my_const_opt_y = float('inf')
    if my_decr_opt_y and my_decr_opt_y > 1000000000:
        my_decr_opt_y = float('inf')
    if my_fast_opt_y and my_fast_opt_y > 1000000000:
        my_fast_opt_y = float('inf')
    if tf_opt_y and tf_opt_y > 1000000000:
        tf_opt_y = float('inf')
    if scipy_opt_y and scipy_opt_y > 1000000000:
        scipy_opt_y = float('inf')

    best = ''
    if min(my_const_opt_y, my_decr_opt_y, my_fast_opt_y, tf_opt_y, scipy_opt_y) == my_const_opt_y:
        best = 'const'
    elif min(my_const_opt_y, my_decr_opt_y, my_fast_opt_y, tf_opt_y, scipy_opt_y) == my_decr_opt_y:
        best = 'decreasing'
    elif min(my_const_opt_y, my_decr_opt_y, my_fast_opt_y, tf_opt_y, scipy_opt_y) == my_fast_opt_y:
        best = 'fastest'
    elif min(my_const_opt_y, my_decr_opt_y, my_fast_opt_y, tf_opt_y, scipy_opt_y) == tf_opt_y:
        best = 'tensorflow'
    elif min(my_const_opt_y, my_decr_opt_y, my_fast_opt_y, tf_opt_y, scipy_opt_y) == scipy_opt_y:
        best = 'scipy'

    # добавление очередной строки
    data.append((x_start, z_start, my_const_opt_y, my_decr_opt_y, my_fast_opt_y, tf_opt_y, scipy_opt_y, best))

table = pd.DataFrame(data, columns=['orig_x', 'orig_z', 'my_const_opt_y', 'my_decr_opt_y', 'my_fast_opt_y', 'tf_opt_y', 'scipy_opt_y', 'best'])
print(table)
print('Best "const" count = ', len(table[(table['best']=='const')]))
print('Best "decreasing" count = ', len(table[(table['best']=='decreasing')]))
print('Best "fastest" count = ', len(table[(table['best']=='fastest')]))
print('Best "tensorflow" count = ', len(table[(table['best']=='tensorflow')]))
print('Best "scipy" count = ', len(table[(table['best']=='scipy')]))
