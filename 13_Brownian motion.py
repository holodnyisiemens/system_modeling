import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter

def get_max_spread(Xi):
    return np.max(np.sqrt(np.sum((Xi ** 2), axis=1)))

# моделирование Броуновского движения
def Brownian_motion_of_particles(V, delta_t, num_points, n_iter, random_state=1):
    # установка зерна для воспроизводимости эксперимента
    rgen = np.random.RandomState(random_state)

    # массив всех значений точек за все время
    X = np.zeros(shape=(n_iter, num_points, 3))

    # массив радиус-векторов на каждой итерации
    P = np.zeros(shape=n_iter)

    # вектор скоростей
    Vxyz = np.empty(shape=3)

    for i in range(1, n_iter):
        for j in range(num_points):
            # генерация случайных величин
            psi1 = rgen.random()
            psi2 = rgen.random()
            psi3 = rgen.random()

            cos_tetta = 2 * psi1 - 1
            sign_sin_tetta = np.sign(2 * psi2 - 1)
    
            if sign_sin_tetta == 0:
                sign_sin_tetta = 1

            sin_tetta = sign_sin_tetta * (1 - cos_tetta ** 2)
    
            fi = 2 * np.pi * psi3

            # изменение скоростей
            Vxyz[0] = V * cos_tetta * np.cos(fi)
            Vxyz[1] = V * cos_tetta * np.sin(fi)
            Vxyz[2] = V * sin_tetta

            # обновленные координаты
            X[i, j, :] = X[i-1, j, :] + delta_t * Vxyz

        # получение максимального радиус-вектора на текущей итерации
        P[i] = get_max_spread(X[i])

    return P, X

# метод наименьших квадратов
def min_squares_method(P, T):
    n = T.shape[0]
    a = (n * np.sum(T * P) - np.sum(T) * np.sum(P)) / (n * np.sum(T ** 2) - (np.sum(T)) ** 2)
    b = (np.sum(P) - a * np.sum(T)) / n
    return a, b
   
# аппроксимация к прямой
def straight_approx(P, T):
    a, b = min_squares_method(P, T)
    P_straight_approx = a * T + b
    return P_straight_approx

# аппроксимация к экспоненте
def exp_approx(P, T, c):
    # переход к аррениусовской системе координат
    P_arrhenius = np.log(P + c)
    a, b = min_squares_method(P_arrhenius, T)
    P_exp_approx = np.exp(a * T + b) - c
    return P_exp_approx

# аппроксимация до прямой с помощью numpy
def numpy_straiht_approx(P, T):
    a, b = np.polyfit(T, P, 1)
    P_straight_approx_np = a * T + b
    return P_straight_approx_np

# аппроксимация до экспоненты с помощью numpy
def numpy_exp_approx(P, T, c):
    P_arrhenius = np.log(P + c)
    a, b = np.polyfit(T, P_arrhenius, 1)
    P_exp_approx_np = np.exp(a * T + b) - c
    return P_exp_approx_np

# получение значений абсолютных погрешностей
def get_errs(approx_vals, true_vals):
    return np.abs(approx_vals - true_vals)

N = 35
V = N
delta_t = 1 / N
num_points = 1000
n_iter = 100

# поправочный параметр при экспоненциальной аппроксимации, чтобы избежать противоречие p(t) != 0
c = 0.0000001

P, X = Brownian_motion_of_particles(V=V, delta_t=delta_t, num_points=num_points, n_iter=n_iter)

T = np.arange(0, n_iter * delta_t, delta_t)

# аппроксимации
P_straight_approx = straight_approx(P, T)
P_exp_approx = exp_approx(P, T, c)

P_straight_approx_np = numpy_straiht_approx(P, T)
P_exp_approx_np = numpy_exp_approx(P, T, c)

P_straight_approx_errs = get_errs(P_straight_approx, P)
P_exp_approx_errs = get_errs(P_exp_approx, P)

P_straight_approx_np_errs = get_errs(P_straight_approx_np, P)
P_exp_approx_np_errs = get_errs(P_exp_approx_np, P)

plt.title('Аппроксимация функции изменения радиус-вектора до прямой')
plt.xlabel('T')
plt.ylabel('p(t)')
plt.plot(T, P_straight_approx_np, label='Аппроксимация до прямой (numpy.polyfit)', linewidth=4)
plt.plot(T, P_straight_approx, label='Аппроксимация до прямой (собственная реализация)', linewidth=2)
plt.plot(T, P, label='Исходная функция')
plt.legend(loc='best')
plt.show()

plt.title('Аппроксимация функции изменения радиус-вектора до экспоненты')
plt.xlabel('T')
plt.ylabel('p(t)')
plt.plot(T, P_exp_approx_np, label='Аппроксимация до экспоненты (numpy.polyfit)', linewidth=4)
plt.plot(T, P_exp_approx, label='Аппроксимация до экспоненты (собственная реализация)', linewidth=2)
plt.plot(T, P, label='Исходная функция')
plt.legend(loc='best')
plt.show()

plt.title('Абсолютные погрешности аппроксимации')
plt.xlabel('T')
plt.ylabel('$\Delta$')
plt.plot(T, P_straight_approx_np_errs, label='При аппроксимации до прямой (numpy.polyfit)', linewidth=4)
plt.plot(T, P_straight_approx_errs, label='При аппроксимации до прямой (собственная реализация)', linewidth=2)
plt.plot(T, P_exp_approx_np_errs, label='При аппроксимации до экспоненты (numpy.polyfit)', linewidth=4)
plt.plot(T, P_exp_approx_errs, label='При аппроксимации до экспоненты (собственная реализация)', linewidth=2)
plt.legend(loc='best')
plt.show()

# список кадров анимации
frames = []

# 3d фигура
fig = plt.figure(figsize=(7, 7))
ax_3d = plt.axes(projection='3d')

for i in range(n_iter):
    # ссылка на текущее изображение (объект-наследник Artist)
    line = ax_3d.scatter3D(X[i, :, 0], X[i, :, 1], X[i, :, 2], color='green')
    frames.append([line])

animation = ArtistAnimation(
    fig,            # фигура, где отображается анимация
    frames,         # набор кадров
    interval=100,   # задержка в мс между кадрами
    repeat=True     # зацикливание анимации
)

plt.show()

# сохранение в GIF
writergif = PillowWriter(fps=5)
animation.save('./animation.gif', writer=writergif)