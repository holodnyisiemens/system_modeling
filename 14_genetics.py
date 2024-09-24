'''
В результате скрещивания двух гетерозиготных по всем признакам 
«родителей» появилось 100, 1000, 1000000 потомков. Провести 
количественный анализ для каждого фенотипа при 1, 2 признаках 
наследования (сопоставить вероятности рождения определенных 
потомков с закономерностями 2-го и 3-го закона Менделя). 
Представить графики зависимости относительной погрешности от 
количества потомков. 

Построить зависимость количества возможных фенотипов от 
количества признаков наследственности в зависимости от количества 
признаков. Аппроксимировать зависимости с использованием 
готовых библиотек'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# установка зерна для воспроизводимости эксперимента
np.random.seed(1)

def get_child_genotype_and_phenotype(parent1, parent2, n):
    child_phenotype = np.empty(shape=(n, parent1.shape[0]), dtype=np.bool_)
    child_genotype = np.empty(shape=(n, parent1.shape[0], 2), dtype=np.bool_)

    for i in range(n):
        # выбор из каждого гена родителей по одной случайной зиготе
        rand_gens_p1 = np.apply_along_axis(func1d=np.random.choice, axis=1, arr=parent1, size=1)
        rand_gens_p2 = np.apply_along_axis(func1d=np.random.choice, axis=1, arr=parent2, size=1)

        # объединение зигот для получения гена потомка
        child_genotype[i, :, :] = np.hstack((rand_gens_p1, rand_gens_p2))

    child_phenotype = np.logical_and(child_genotype[:, :, 0], child_genotype[:, :, 1])

    return child_genotype, child_phenotype

def get_separation_by_phenotype(child_phenotype):
    # уникальные комбинации по фенотипу
    possible_phenotypes = np.unique(child_phenotype, axis=0)

    # число комбинаций
    phenotypes_num = possible_phenotypes.shape[0]

    separation_by_phenotype = np.empty(shape=phenotypes_num)

    # расчет разделения по фенотипу
    for i in range(phenotypes_num):
        separation_by_phenotype[i] = np.count_nonzero(np.all(child_phenotype == possible_phenotypes[i], axis=1))

    return separation_by_phenotype, phenotypes_num

# получение значений относительных погрешностей погрешностей
def get_errs(calc_vals, true_vals):
    return np.abs((calc_vals - true_vals) / true_vals)

# аппроксимация до экспоненты с помощью numpy
def numpy_exp_approx(X, Y):
    Y_arrhenius = np.log(Y)
    a, b = np.polyfit(X, Y_arrhenius, 1)
    Y_exp_approx_np = np.exp(a * X + b)
    return Y_exp_approx_np

# число потомков
child_numbers = np.array([100, 1000, 10000, 100000, 1000000])

# вероятности рождения каждого из потомков по второму закону Менделя
one_trait_mendel_probabilities = np.array([3/4, 1/4])

# разделение по фенотипу при 1 наследуемом признаке (должно приближаться к 3:1 по второму закону Менделя)
one_trait_phenotype_proportions = np.empty(shape=(child_numbers.shape[0], 2))

# расчитываемые вероятности каждом количестве потомков для особей с 1 признаком
one_trait_calc_probabilities = np.empty(shape=(child_numbers.shape[0], 2))

for i in range(child_numbers.shape[0]):
    # 2 гетерозиготных по единственному признаку родителя
    parent1 = np.array([[1, 0]])
    parent2 = np.array([[1, 0]])

    child_genotype, child_phenotype = get_child_genotype_and_phenotype(parent1, parent2, child_numbers[i])
    separation_by_phenotype, phenotypes_num = get_separation_by_phenotype(child_phenotype)

    # доля рожденных для каждого случая от общего числа рожденных особей
    calc_probabilities = separation_by_phenotype / child_numbers[i]

    one_trait_calc_probabilities[i] = calc_probabilities

    # 4 комбинации, т.к. известно, что особи гетерозиготные по единственному признаку
    phenotype_proportion = calc_probabilities * 4

    one_trait_phenotype_proportions[i] = phenotype_proportion

# относительные погрешности вероятностей рождения особи с определенным(и) признак(ом/ами) в зависимости от количества потомков
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
fig.suptitle('Графики относительных погрешностей вероятности рождения потомков\nот числа потомков (относительно 2 закона Менделя)')

axs[0].plot(np.log10(child_numbers), get_errs(one_trait_calc_probabilities[:,0], one_trait_mendel_probabilities[0]), label='Для рождения особи с наследуемым доминатным геном')
axs[1].plot(np.log10(child_numbers), get_errs(one_trait_calc_probabilities[:,1], one_trait_mendel_probabilities[1]), label='Для рождения особи с наследуемым рецессивным геном')

for i in range(2):
    axs[i].legend(loc='best', fontsize=8)
    plt.xlabel('Число потомков')
    plt.ylabel('Относительная погрешность вероятности рождения')

# настройка расстояния между подграфиками
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()

# расчитываемые вероятности каждом количестве потомков для особей с 2 признаками
two_trait_calc_probabilities = np.empty(shape=(child_numbers.shape[0], 4))

# разделение по фенотипу при 2 наследуемых признаках (должно приближаться к 9:3:3:1 по третьему закону Менделя)
two_trait_phenotype_proportions = np.empty(shape=(child_numbers.shape[0], 4))

# вероятности рождения каждого из потомков по третьему закону Менделя
two_trait_mendel_probabilities = np.array([9/16, 3/16, 3/16, 1/16])

for i in range(child_numbers.shape[0]):
    # 2 гетерозиготных по единственному признаку родителя
    parent1 = np.array([[1, 0], [1, 0]])
    parent2 = np.array([[1, 0], [1, 0]])

    child_genotype, child_phenotype = get_child_genotype_and_phenotype(parent1, parent2, child_numbers[i])
    separation_by_phenotype, phenotypes_num = get_separation_by_phenotype(child_phenotype)

    # доля рожденных для каждого случая от общего числа рожденных особей
    calc_probabilities = separation_by_phenotype / child_numbers[i]

    two_trait_calc_probabilities[i] = calc_probabilities

    # 16 комбинаций, т.к. известно, что особи гетерозиготные по каждому из двух признаков
    phenotype_proportion = calc_probabilities * 16

    two_trait_phenotype_proportions[i] = phenotype_proportion

# относительные погрешности вероятностей рождения особи с определенным(и) признак(ом/ами) в зависимости от количества потомков
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.suptitle('Графики относительных погрешностей вероятности рождения потомков\nот числа потомков (относительно 3 закона Менделя)')

axs[0, 0].plot(np.log10(child_numbers), get_errs(two_trait_calc_probabilities[:,0], two_trait_mendel_probabilities[0]), label='Для рождения особи с доминатными генами')
axs[0, 1].plot(np.log10(child_numbers), get_errs(two_trait_calc_probabilities[:,1], two_trait_mendel_probabilities[1]), label='Для рождения особи с доминантным и\nрецессивным генами')
axs[1, 0].plot(np.log10(child_numbers), get_errs(two_trait_calc_probabilities[:,2], two_trait_mendel_probabilities[2]), label='Для рождения особи с доминантным и\nрецессивным генами')
axs[1, 1].plot(np.log10(child_numbers), get_errs(two_trait_calc_probabilities[:,3], two_trait_mendel_probabilities[3]), label='Для рождения особи с рецессивными генами')

for i in range(2):
    for j in range(2):
        axs[i, j].legend(loc='best', fontsize=8)

# настройка расстояния между подграфиками
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()

n = 1000

# количество признаков наследственности
trait_nums = np.arange(1, 10)
phenotypes_nums = np.empty_like(trait_nums)

# ген из разных зигот
gene = np.array([[1, 0]])

for t_num in trait_nums:
    # создание нужного числа генов для разного числа признаков
    parent1 = np.concatenate([gene]  *  t_num)
    parent2 = parent1

    _, child_phenotype = get_child_genotype_and_phenotype(parent1, parent2, n)
    _, p_num = get_separation_by_phenotype(child_phenotype)

    phenotypes_nums[t_num - 1] = p_num

exp_approx_phenotypes_nums = numpy_exp_approx(trait_nums, phenotypes_nums)

plt.title(f'Изменение числа возможных фенотипов (для {n} потомков)')
plt.plot(trait_nums, phenotypes_nums, label='Исходная функция')
plt.plot(trait_nums, exp_approx_phenotypes_nums, label='Аппроксимация до экспоненты')
plt.xlabel('Число признаков наследственности')
plt.ylabel('Число возможных фенотипов')
plt.legend(loc='best')
plt.show()

# таблица с разделениями по фенотипу
data = []

for n, p1, p2 in zip(child_numbers, one_trait_phenotype_proportions, two_trait_phenotype_proportions):
    data.append((n, p1, p2))

table = pd.DataFrame(data, columns=['Число потомков', 'Разделение по фенотипу при одном признаке', 'Разделение по фенотипу при двух признаках'])
print(table)
