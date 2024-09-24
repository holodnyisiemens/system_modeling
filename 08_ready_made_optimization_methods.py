from scipy.optimize import minimize
import pandas as pd
import random
import numpy as np
import time
import tensorflow as tf
# отключение сообщений об ошибках при работе TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# исходная функция, которую нужно оптимизировать
def func(vals: np.array) -> float:
    x, z = vals
    return np.sin(z % (2*np.pi)) + np.exp(-x)

'''Многомерный метод Ньютона'''

# первая частная производная y по x
def get_first_part_deriv_y_on_x(x: float) -> float:
    return -np.exp(-x)

# первая частная производная y по z
def get_first_part_deriv_y_on_z(z: float) -> float:
    # отброс периодической части
    return np.cos(z % (2*np.pi))

# получение градиента (вектора частных производных)
def get_gradient(vals: np.array):
    x, z = vals
    return np.array((get_first_part_deriv_y_on_x(x), get_first_part_deriv_y_on_z(z)))

# вычисление Гессиана
def get_hessian(vals: np.array):
    x, z = vals
    H = np.zeros_like(vals)
    H[0] = np.e ** -x
    # отброс периодической части
    H[-1] = -np.sin(z % (2*np.pi))
    return np.diag(H)

def newton_optimize(x: float, z: float, acc: float) -> tuple:
    vals = np.array((x, z))
    # параметр jac - функция вычисления градиента, hess - функция, возвращающая матрицу Гесса
    res = minimize(func, vals, method='Newton-CG', jac=get_gradient, hess=get_hessian, options={'xtol': acc})
    return res.x[0], res.x[1], res.fun

'''Метод градиентного спуска'''

# исходная функция для оптимизации с помощью Tensorflow
def func_tf(x: tf.Variable, z: tf.Variable):
    return tf.sin(z) + tf.exp(-x)

def grad_descent_optimize(x: float, z: float, acc: float):
    # преобразование для Tensorflow
    x = tf.Variable(float(x), dtype=tf.float64)
    z = tf.Variable(float(z), dtype=tf.float64)
    learning_rate = 0.01
    for _ in range(1000):
        with tf.GradientTape() as tape:
            loss = func_tf(x, z)
        gradient = tape.gradient(loss, [x, z])
        x.assign_sub(learning_rate * gradient[0])
        z.assign_sub(learning_rate * gradient[1])

    # обратное преобразование
    x = float(x)
    z = float(z)
    vals = np.array((x, z))
    return x, z, func(vals)

'''Метод сопряженных градиентов'''

def conjugate_grad_optimize(x: float, z: float, acc: float) -> tuple:
    vals = np.array((x, z))
    # параметр jac - функция вычисления градиента, hess - функция, возвращающая матрицу Гесса
    res = minimize(func, vals, method='CG', jac=get_gradient)
    return res.x[0], res.x[1], res.fun

'''Метод Нелдера-Мида'''

def nelder_mead_optimize(x_start: float, z_start: float, acc: float) -> tuple:
    vals = np.array((x_start, z_start))
    res = minimize(func, vals, method='nelder-mead')
    return res.x[0], res.x[1], res.fun

# 30 тестов
# случайные начальные значения x и z
x_data = [random.randint(-100, 100) for _ in range(31)]
z_data = [random.randint(-100, 100) for _ in range(31)]

# шаг сходимости
lamb = 0.01
# точность
acc = 1e-6
# шаг формирования симплекса относительно исходной точки с ранее описанными случайными значениями x, z, а также вычисленным по ним y
simplex_step = 0.01

# таблица с результатами
data = []

for x_start, z_start in zip(x_data, z_data):
    vals = np.array((x_start, z_start))
    y_start = func(vals)

    start_time = time.perf_counter_ns()
    newton_opt_x, newton_opt_z, newton_opt_y = newton_optimize(x=x_start, z=z_start, acc=acc)
    end_time = time.perf_counter_ns()
    newton_time = (end_time - start_time) / 1e6

    start_time = time.perf_counter_ns()
    grad_desc_opt_x, grad_desc_opt_z, grad_desc_opt_y = grad_descent_optimize(x=x_start, z=z_start, acc=acc)
    end_time = time.perf_counter_ns()
    grad_desc_time = (end_time - start_time) / 1e6

    start_time = time.perf_counter_ns()
    conj_grad_opt_x, conj_grad_opt_z, conj_grad_opt_y = conjugate_grad_optimize(x=x_start, z=z_start, acc=acc)
    end_time = time.perf_counter_ns()
    conj_grad_time = (end_time - start_time) / 1e6

    start_time = time.perf_counter_ns()
    nelder_mead_opt_x, nelder_mead_opt_z, nelder_mead_opt_y = nelder_mead_optimize(x_start=x_start, z_start=z_start, acc=acc)
    end_time = time.perf_counter_ns()
    nelder_mead_time = (end_time - start_time) / 1e6

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
