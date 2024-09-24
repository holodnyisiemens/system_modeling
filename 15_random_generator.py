'''Генерация нормальной случайной величины с помощью преобразования Бокса-Мюллера'''

import numpy as np
import matplotlib.pyplot as plt

# функция плотности нормального распределения
def normal_distribution_probability_density(X, mean, std_dev):
    return 1 / (np.sqrt(2 * np.pi) * std_dev) * np.exp(-(X - mean) ** 2 / (2 * (std_dev ** 2)))

def get_normal_distribution_numpy(N, random_state):
    rgen = np.random.RandomState(random_state)
    X_normal_numpy = rgen.normal(size=N)
    return X_normal_numpy

# N - количество генерируемых значений
# генерация нормальной случайной величины с пмощью преобразования Бокса-Мюллера
def get_normal_distribution(N, random_state):
    # установка зерна для воспроизводимости эксперимента
    rgen = np.random.RandomState(random_state)

    # массив случаных величин, сводящихся к нормальным
    X_normal = np.empty(shape=N)

    for i in range(N):
        # равномерно распределенные величины от 0 до 1
        psi1 = rgen.random()
        psi2 = rgen.random()

        # случайная координата на окружности (радиус и угол)
        r = psi1
        a = 2 * np.pi * psi2

        # случайная величина (сводится к нормальному распределению)
        x = np.sqrt(-2 * np.log(r)) * np.cos(a)
        # или y = np.sqrt(-2 * np.log(r)) * np.sin(a)

        X_normal[i] = x

    return X_normal

# нормировка СВ (L - максимальное значение СВ)
def normalize_random_vals(rand_vals, L):
    max_val = np.max(rand_vals)
    min_val = np.min(rand_vals)

    # масштабирование значений СВ (от 0 до L)
    mean = L / 2
    std_dev = 2 / (max_val - min_val)

    # конечное представление СВ с учетом масштабирования (значения от 0 до L)
    rand_vals_std = mean + std_dev * rand_vals

    return rand_vals_std, mean, std_dev    

def get_occurrences_number(X_normal_std_sorted, subbarr_borders):
    # массив числа вхождений в поддиапазоны
    k = np.empty(shape=subbarr_borders.shape[0] - 1)

    # подсчет количества вхождений чисел в поддиапазоны
    for i in range(subbarr_borders.shape[0] - 1):
        k[i] = np.count_nonzero((subbarr_borders[i] <= X_normal_std_sorted) & (X_normal_std_sorted < subbarr_borders[i+1]))

    return k

# модифицированный метод Эйлера с пересчетом
def modified_Euler_method(x0, y0, f, h):
    n = f.shape[0]
    # x0, y0 - начальные условия
    # h - шаг = xi+1 - xi
    # n - количество шагов
    Y = np.empty(shape=(n,))
    Y[0] = y0
    xi = x0
    yi = y0

    for i in range(1, n - 1):
        # не используется, т.к. функция не зависит от y
        # predict_yi = yi + f(xi) * h
        corrected_yi = yi + (f[i] + f[i + 1]) / 2 * h
        yi = corrected_yi
        xi += h
        Y[i] = yi

    return Y

# количества генерируемых случайных величин (СВ)
N_arr = [10, 100, 1000, 10000]

for N in N_arr:
    # нормирующее число
    L = 35

    # количество поддиапазонов, на которые разбивается диапазон всех значений
    subarrays_num = 100

    # массив поддиапазонов возможных значений
    subbarr_borders = np.linspace(0, L, subarrays_num + 1)

    # количество раз, сколько будут генерироваться случайные значения
    generate_number = 100

    # двумерный массив, где каждая строка - зависимость количества СВ, попавших в каждый элемент, поделенных на полное кол-во сгенерированных СВ
    f_empirical_examples = np.empty(shape=(generate_number, subarrays_num))

    f_empirical_examples_numpy = np.empty(shape=(generate_number, subarrays_num))

    for i in range(generate_number):
        # зерно изменяется с каждой итерацией
        X_normal= get_normal_distribution(N, random_state=i)
        X_normal_numpy = get_normal_distribution_numpy(N, random_state=i)

        X_normal_std, mean, std_dev = normalize_random_vals(X_normal, L)
        X_normal_std_numpy, mean_numpy, std_dev_numpy = normalize_random_vals(X_normal_numpy, L)

        # сортировка
        X_normal_std_sorted = np.sort(X_normal_std)
        X_normal_std_sorted_numpy = np.sort(X_normal_std_numpy)

        # массив числа вхождений в поддиапазоны
        k = get_occurrences_number(X_normal_std_sorted, subbarr_borders)
        k_numpy = get_occurrences_number(X_normal_std_sorted_numpy, subbarr_borders)

        # эмпирическое представление плотности вероятности
        # L/subarrays_num - длина поддиапазона
        f_empirical_examples[i, :] = subarrays_num * k / (N * L)
        f_empirical_examples_numpy[i, :] = subarrays_num * k_numpy / (N * L)

    # шаг для построения эмпирических функций
    step_empirical = subbarr_borders[1] - subbarr_borders[0]

    # шаг для построения аналитических функций
    step_theoretical = 0.01

    X_arange = np.arange(0, L, step_theoretical)

    # FIXME разное количество генерируемых СВ

    # средние значения функций плотности относительно многократно сгенерированных случайных величин
    f_empirical_average = np.average(f_empirical_examples, axis=0)
    f_empirical_average_numpy = np.average(f_empirical_examples_numpy, axis=0)

    # теоретическое представление плотности вероятности
    f_theoretical = normal_distribution_probability_density(X_arange, mean, std_dev)

    # зависимость вероятности от x при теоретическом и эмпирическом представлениях плотности вероятности
    # для каждого x равняется интегралу по функции плотности от минус бесконечности до x (x - СВ)
    # первообразная функции плотности - функция распределения СВ
    # в примере с L=35 функция распределения в точке 0 приблизительно равна 0
    x0 = 0
    y0 = 0

    p_empirical = modified_Euler_method(x0, y0, f_empirical_average, h=step_empirical)
    p_empirical_numpy = modified_Euler_method(x0, y0, f_empirical_average_numpy, h=step_empirical)
    p_theoretical = modified_Euler_method(x0, y0, f_theoretical, h=step_theoretical)

    # при стремлении к бесконечности аргумента функции распределения ее значение стремится к 1
    p_empirical[-1] = 1
    p_empirical_numpy[-1] = 1
    p_theoretical[-1] = 1

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    fig.suptitle(f'Графики плотности и функции распределения нормального распределения СВ при {N} генерируемых значений')

    axs[0].plot(X_arange, f_theoretical, linewidth=3, label='Аналит.')
    axs[0].plot(subbarr_borders[:-1], f_empirical_average_numpy, linewidth=3, label='Эмп. np.random.normal')
    axs[0].plot(subbarr_borders[:-1], f_empirical_average, linewidth=1, label='Эмп. Бокса-Мюллера')
    axs[0].legend(loc='upper left')

    axs[1].plot(X_arange, p_theoretical, linewidth=3, label='Аналит.')
    axs[1].plot(subbarr_borders[:-1], p_empirical_numpy, linewidth=3, label='Эмп. np.random.normal')
    axs[1].plot(subbarr_borders[:-1], p_empirical, linewidth=1, label='Эмп. Бокса-Мюллера')
    axs[1].legend(loc='upper left')

    plt.show()
