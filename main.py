from models import Magnet, Shaker, Coil
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import logging
from datetime import datetime
import os

# Проверка и создание директории для логов
if not os.path.exists('logs'):
    os.makedirs('logs')

# Настройка логгера
logger = logging.getLogger('magnet_simulation')
logger.setLevel(logging.DEBUG)
timestamp = datetime.now().strftime('%d.%m.%Y-%H-%M-%S')

# Создаем обработчик для логирования в файл
fh = logging.FileHandler(f'logs/{timestamp}.log')
fh.setLevel(logging.DEBUG)

# Создаем обработчик для вывода логов в консоль
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Настройка форматирования логов
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Добавляем обработчики к логгеру
logger.addHandler(fh)
logger.addHandler(ch)

results = []
eds_forces = []

def get_magnet_position(shaker, t):
    return shaker.X0 * np.sin(shaker.W * t)

def calculate_f_damping(v_m, magnet):
    """
    Расчет силы демпфирования (сопротивления воздуха).
    """
    Cd = 1.2  # Коэффициент лобового сопротивления (для турбулентного потока)
    ro = 1.225  # Плотность воздуха (кг/м^3)
    A = math.pi * magnet.diameter ** 2 * 0.25  # Площадь поперечного сечения магнита
    F_damping = 0.5 * ro * v_m ** 2 * Cd * A * np.sign(v_m)
    return F_damping


def calculate_f_air(z_m, v_m, magnet, z_top, z_bottom):
    """
    Расчет силы вязкого трения воздуха в зазоре между магнитом и цилиндром.
    """
    # Диаметры внутреннего цилиндра и магнита
    D_outer = 0.0196  # Диаметр внутренней стенки цилиндра (м)
    D_inner = magnet.diameter  # Диаметр магнита (м)

    # Зазор между магнитом и цилиндром
    gap = (D_outer - D_inner) / 2  # Радиальный зазор (м)

    if gap <= 0:
        raise ValueError("Диаметр магнита должен быть меньше диаметра цилиндра.")

    # Параметры воздуха
    mu_air = 1.81e-5  # Динамическая вязкость воздуха (Па·с)

    # Сила вязкого трения для потока в кольцевом зазоре:
    F_viscous = -6 * np.pi * mu_air * magnet.height * v_m / gap
    # Сила вязкого трения для цилиндра в цилиндре:
    ln_ratio = np.log(D_outer / D_inner)
    # F_viscous = - (4 * np.pi * mu_air * magnet.height * v_m) / ln_ratio

    return F_viscous


def combined_equations(t, y, magnet, shaker, z_top, z_bottom):
    """
    Система дифференциальных уравнений для расчета движения магнита.
    """
    z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk = y

    # Константы и параметры
    F_gravity = magnet.mass * shaker.G

    # Силы
    F_shaker = shaker.get_force(magnet, t)
    a_sk = F_shaker / magnet.mass  # Ускорение шейкера

    F_top_magnetic = magnet.get_force(
        abs(z_m - (magnet.height / 2) - z_top) + get_magnet_position(shaker, t)
    )
    F_bottom_magnetic = magnet.get_force(
        abs(z_m - (magnet.height / 2) - z_bottom) + get_magnet_position(shaker, t)
    )

    # Расчет силы демпфирования
    F_damping = calculate_f_damping(v_m, magnet)

    # Расчет силы от вязкого трения воздуха
    F_viscous = calculate_f_air(z_m, v_m, magnet, z_top, z_bottom)

    # Общая сила на магнит
    F_total_magnet = (
        - F_top_magnetic
        + F_bottom_magnetic
        - F_gravity
        - F_damping 
        + F_viscous
    )

    # Логирование (опционально)
    logger.info(f"F total: {F_total_magnet}")
    logger.info(f"F_top_magnetic: {-F_top_magnetic}")
    logger.info(f"F_bottom_magnetic: {F_bottom_magnetic}")
    logger.info(f"F_gravity: {-F_gravity}")
    logger.info(f"F_shaker: {F_shaker}")
    logger.info(f"F_damping: {F_damping}")
    logger.info(f"F_viscous: {F_viscous}")
    logger.info(f"Z {z_m}")
    logger.info(f"Time {t}\n---------------------------------------------")

    # Вычисление ускорений
    a_m = F_total_magnet / magnet.mass  # Ускорение магнита
    a_tm = F_shaker / magnet.mass  # Ускорение верхнего магнита
    a_bm = F_shaker / magnet.mass  # Ускорение нижнего магнита

    # Возврат производных переменных состояния
    return [v_m, a_m, v_tm, a_tm, v_bm, a_bm, v_sk, a_sk]


def save_eds_forces(z_m, v_m, t, a, shaker, coil):
    """
    Функция для сохранения значений ЭДС и результатов.
    """
    eds_per_turn, total_eds = coil.get_eds(shaker, z_m, v_m, t, a)
    eds_forces.append(total_eds)

    results.append(
        {
            't': t,
            'z_m': z_m,
            'v_m': v_m,
            'eds': total_eds,
        }
    )


def main():
    # Создание объекта магнита
    magnet = Magnet(
        diameter=0.0195,  # Диаметр магнита 19.5 мм
        mass=0.043,       # Масса магнита 43 г
        height=0.01,      # Высота магнита 10 мм
    )

    # Позиция магнитов
    z_top = 0.07
    z_bottom = 0.01
    G = 9.8  # Ускорение свободного падения (м/с^2)
    X0 = 0.001  # Амплитуда колебаний
    μ = 50     # Частота колебаний
    time_total = 2  # Время моделирования
    magnet_start_z = 0.03
    shaker = Shaker(
        G=G,
        miew=μ,
        X0=X0,
    )

    coil = Coil(
        turns_count=208,
        thickness=0.01025,
        radius=0.01025,
        position=0.015,
        magnet=magnet,
        wire_diameter=0.000961538462,
        layer_count=4,
    )

    # Начальные условия: [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk]
    initial_conditions = [magnet_start_z, 0, z_top, 0, z_bottom, 0, 0.0, 0]
    t_span = (0, time_total)
    t_eval = np.linspace(0, time_total, 10000)

    # Решение системы уравнений
    # Решение системы уравнений
    sol_combined = solve_ivp(
        combined_equations,
        t_span,
        initial_conditions,
        args=(magnet, shaker, z_top, z_bottom),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-6,
        dense_output=True,  # Добавьте этот параметр
    )


    # Получаем значения решения на точках `t_eval`
    solution_values = sol_combined.sol(t_eval)
    a_values = np.gradient(solution_values[1], sol_combined.t)  # Используем градиент для численного дифференцирования

    # Рассчитываем ЭДС на каждом шаге
    for i, t in enumerate(t_eval):
        z_m = solution_values[0, i]
        v_m = solution_values[1, i]
        a_m = a_values[i]

        save_eds_forces(z_m, v_m, t, a_m, shaker, coil)

    # Построение графиков
    plt.figure(figsize=(12, 12))

    # Положение магнита и шейкера
    plt.subplot(4, 1, 1)
    plt.plot(sol_combined.t, sol_combined.y[0], label='Магнит (z)', color='blue')
    plt.plot(sol_combined.t, sol_combined.y[6], label='Шейкер (z)', color='green')
    plt.plot(sol_combined.t, sol_combined.y[2], label='Верхний магнит (z)', color='purple')
    plt.plot(sol_combined.t, sol_combined.y[4], label='Нижний магнит (z)', color='orange')
    plt.plot(
        sol_combined.t,
        [coil.position for _ in sol_combined.t],
        label='Катушка нижняя граница',
        color='red',
    )
    plt.plot(
        sol_combined.t,
        [(coil.position + coil.height) for _ in sol_combined.t],
        label='Катушка верхняя граница',
        color='red',
    )
    plt.xlabel('Время (с)')
    plt.ylabel('Положение (м)')
    plt.legend()
    plt.grid()

    # ЭДС
    plt.subplot(4, 1, 2)
    plt.plot(
        sol_combined.t,
        eds_forces,
        color='red',
        label=f"Сила ЭДС\nКоличество витков: {coil.turns_count}\n"
        f"Количество витков на слой: {coil.turns_per_layer}\n"
        f"Количество слоёв: {coil.layer_count}\n"
        f"Высота катушки: {coil.height}",
    )
    plt.xlabel('Время (с)')
    plt.ylabel('ЭДС (В)')
    plt.legend()
    plt.grid()

    # Скорость магнита
    plt.subplot(4, 1, 3)
    plt.plot(
        sol_combined.t,
        sol_combined.y[1],
        label='Скорость магнита (v)',
        color='cyan',
    )
    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с)')
    plt.legend()
    plt.grid()

    # Сила магнита
    f_values = a_values * magnet.mass
    plt.subplot(4, 1, 4)
    plt.plot(
        sol_combined.t,
        f_values,
        label='Сила магнита',
        color='orange',
    )
    plt.xlabel('Время (с)')
    plt.ylabel('Сила (Н)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show() 

if __name__ == '__main__':
    main()
    logger.info('----------------END----------------')
