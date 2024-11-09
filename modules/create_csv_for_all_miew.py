from models import Magnet, Shaker, Coil
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import csv 
from datetime import datetime 
import os
import re
 
results = []
eds_forces = [] 
csv_dir = "exp_csv/a1"


def extract_number(file_name):
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]) if numbers else float('inf')  # Default to infinity if no number is found

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
            'eds': total_eds,
        }
    )

def create_csv(miew):
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
        miew=miew,
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
    for i, t in enumerate(t_eval):
        z_m = solution_values[0, i]
        v_m = solution_values[1, i]
        a_m = a_values[i]

        save_eds_forces(z_m, v_m, t, a_m, shaker, coil)

 

if __name__ == '__main__': 
    miews = sorted([f.split('.')[0] for f in os.listdir(csv_dir) if f.endswith('.csv')], key=extract_number)

    for miew in miews:
        print(f"Creating for μ={miew}")
        create_csv(int(miew))
        with open(f'csv/a1/{miew}.csv', mode='w', newline='') as csv_file:
            fieldnames = ['t', 'eds']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for data in results:
                writer.writerow(data) 