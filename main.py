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
# logger.setLevel(logging.DEBUG)
# timestamp = datetime.now().strftime('%d.%m.%Y-%H-%M-%S')

# # Создаем обработчик для логирования в файл
# fh = logging.FileHandler(f'logs/{timestamp}.log')
# fh.setLevel(logging.DEBUG)

# # Создаем обработчик для вывода логов в консоль
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# # Настройка форматирования логов
# formatter = logging.Formatter(
#     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# # Добавляем обработчики к логгеру
# logger.addHandler(fh)
# logger.addHandler(ch)

def get_magnet_position(shaker, t):
    return shaker.X0 * np.sin(shaker.W * t)

def calculate_f_damping(v_m, magnet):
    """
    Расчет силы демпфирования (сопротивления воздуха).
    """
    Cd = 1.2  # Коэффициент лобового сопротивления (для турбулентного потока)
    ro = 1.225  # Плотность воздуха (кг/м^3)
    A = math.pi * magnet.diameter ** 2 * 0.25  # Площадь поперечного сечения магнита
    F_damping = 0.5 * ro * v_m ** 2 * Cd * np.sign(v_m)
    return F_damping


def calculate_f_air(z_m, v_m, magnet, z_top, z_bottom):
    """
    Расчет силы вязкого трения воздуха в зазоре между магнитом и цилиндром.
    """
    # Диаметры внутреннего цилиндра и магнита
    D_outer = 0.0205  # Диаметр внутренней стенки цилиндра (м)
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
    # ln_ratio = np.log(D_outer / D_inner)
    # F_viscous = - (4 * np.pi * mu_air * magnet.height * v_m) / ln_ratio

    return F_viscous


def combined_equations(t, y, magnet, shaker, z_top, z_bottom, coil, resistance):
    """
    Система дифференциальных уравнений для расчета движения магнита и тока в катушке.
    """
    z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk, i = y  # Добавили ток i в переменные состояния

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

    # Вычисление ускорений
    a_m = F_total_magnet / magnet.mass  # Ускорение магнита
    a_tm = F_shaker / magnet.mass       # Ускорение верхнего магнита
    a_bm = F_shaker / magnet.mass       # Ускорение нижнего магнита

    # Расчет ЭДС
    eds_per_turn, total_eds = coil.get_total_emf(shaker, z_m, v_m, t, a_m)

    # Индуктивность катушки
    inductance = coil.calculate_inductance()

    # Дифференциальное уравнение для тока в катушке
    di_dt = (total_eds - resistance * i) / inductance

    # Возврат производных переменных состояния
    return [v_m, a_m, v_tm, a_tm, v_bm, a_bm, v_sk, a_sk, di_dt]


def main():
    # Создание объекта магнита
    magnet = Magnet(
        diameter=0.0195,  # Диаметр магнита 19.5 мм
        mass=0.043,       # Масса магнита 43 г
        height=0.01,      # Высота магнита 10 мм
    )

    # Позиция магнитов
    z_top = 0.2
    z_bottom = 0.01
    G = 9.8  # Ускорение свободного падения (м/с^2)
    X0 = 0.001  # Амплитуда колебаний
    μ = 5     # Частота колебаний
    time_total = 2  # Время моделирования
    magnet_start_z = 0.025
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

    resistance = 0.1  # Сопротивление катушки

    # Начальные условия: [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk, i]
    initial_conditions = [magnet_start_z, 0, z_top, 0, z_bottom, 0, 0.0, 0, 0]  # Добавили ток i = 0
    t_span = (0, time_total)
    t_eval = np.linspace(0, time_total, 5000)

    # Решение системы уравнений
    sol_combined = solve_ivp(
        combined_equations,
        t_span,
        initial_conditions,
        args=(magnet, shaker, z_top, z_bottom, coil, resistance),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-6,
    )

    # Получаем значения решения на точках `t_eval`
    solution_values = sol_combined.y

    # Достаем значения переменных
    z_m_values = solution_values[0]
    v_m_values = solution_values[1]
    i_values = solution_values[8]  # Ток в катушке

    # Рассчитываем ускорение магнита
    a_m_values = np.gradient(v_m_values, sol_combined.t)

    # Рассчитываем ЭДС на каждом шаге
    total_eds_values = []
    for i in range(len(t_eval)):
        t = t_eval[i]
        z_m = z_m_values[i]
        v_m = v_m_values[i]
        a_m = a_m_values[i]
        eds_per_turn, total_eds = coil.get_total_emf(shaker, z_m, v_m, t, a_m)
        total_eds_values.append(total_eds)

    total_eds_values = np.array(total_eds_values)

    # Рассчитываем ЭДС самоиндукции на каждом шаге
    inductance = coil.calculate_inductance()
    emf_self_induction = -inductance * np.gradient(i_values, sol_combined.t)

    # Итоговая ЭДС с учетом самоиндукции
    total_emf_with_self_induction = total_eds_values + emf_self_induction

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
        total_emf_with_self_induction,
        color='red',
        label='Итоговая ЭДС с учетом самоиндукции',
    )
    plt.xlabel('Время (с)')
    plt.ylabel('ЭДС (В)')
    plt.legend()
    plt.grid()

    # Внешняя ЭДС и ЭДС самоиндукции
    plt.subplot(4, 1, 3)
    plt.plot(
        sol_combined.t,
        total_eds_values,
        label='Внешняя ЭДС (total_eds)',
        color='blue',
    )
    plt.plot(
        sol_combined.t,
        emf_self_induction,
        label='ЭДС самоиндукции (self_emf)',
        color='green',
    )
    plt.xlabel('Время (с)')
    plt.ylabel('ЭДС (В)')
    plt.legend()
    plt.grid()

    # Ток в катушке
    plt.subplot(4, 1, 4)
    plt.plot(
        sol_combined.t,
        i_values,
        label='Ток в катушке (i)',
        color='orange',
    )
    plt.xlabel('Время (с)')
    plt.ylabel('Ток (А)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'saved_h_{z_top}_{μ}.png')
    plt.show() 

if __name__ == '__main__': 
    main() 

    logger.info('----------------END----------------') 
