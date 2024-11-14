from models.models_v2 import Magnet, Shaker, Coil
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
csv_dir = "exp_csv/a2"


def extract_number(file_name):
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]) if numbers else float('inf')  # Default to infinity if no number is found

def create_csv(miew):
    # Создание объекта магнита
    magnet = Magnet(
        diameter=0.0195,  # Диаметр магнита 19.5 (мм) 
        mass=0.043,       # Масса магнита    43 (г)
        height=0.01       # Высота магнита  10 (мм)
    )

    # Позиция магнитов
    z_top = 0.06
    z_bottom = 0.01
    G = 9.8 # ускорение свободного падения (м/с^2) г. Алматы
    X0 = 0.002 # амплитуда колебаний 

    shaker = Shaker(
        G=9.8,
        miew=miew,
        X0=X0
        )

    coil = Coil(
        turns_count=208,                # Уменьшенное количество витков
        thickness=0.01025,       # Толщина катушки равна радиусу 1.025 мм
        radius=0.01025,          # Радиус катушки 1.025 мм
        position=0.023,
        magnet=magnet,
        wire_diameter=0.0005,      # Диаметр провода 0.5 мм
        layer_count=4
    )   

    # Список для хранения силы ЭДС
    eds_forces = []

    # Define combined equations with ЭДС
    def combined_equations(t, y):
        z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk = y

        # Константы и параметры
        F_gravity = magnet.mass * G

        # Силы
        F_shaker = shaker.get_force(magnet, t)
        a_sk = F_shaker / magnet.mass  # Ускорение шейкера

        F_top_magnetic = magnet.get_force(abs(z_m-(magnet.height/2) - z_top), F_shaker) 
        F_bottom_magnetic = magnet.get_force(abs(z_m-(magnet.height/2) - z_bottom), F_shaker)

        # Коэффициент сопротивления 
        Cd = 1.2  # Турбулентный поток
        ro = 1.225 
        A = math.pi * magnet.diameter**2 * 0.25
        F_damping = 0.5 * ro * v_m**2 * Cd * A
        

        # Корректировка сил магнитов (отталкивание)
        # F_top_magnetic = max(F_top_magnetic, 0)
        # F_bottom_magnetic = max(F_bottom_magnetic, 0)

        # Общая сила на главный магнит
        F_total_magnet = -F_top_magnetic + F_bottom_magnetic - F_gravity - F_damping

 
        # Вычисление ускорений
        a_m = F_total_magnet / magnet.mass  # Ускорение главного магнита
        a_tm = F_shaker / magnet.mass   # Ускорение верхнего магнита
        a_bm = F_shaker / magnet.mass  # Ускорение нижнего магнита

        # Возврат производных переменных состояния
        return [v_m, a_m, v_tm, a_tm, v_bm, a_bm, v_sk, a_sk]


    # Обратный вызов для записи значения ЭДС в моменты `t_eval`
    def save_eds_forces(z_m, v_m, t):
        eds_per_turn, total_eds = coil.get_eds(shaker, z_m, v_m, t)
        # print('EDS:', eds)
        eds_forces.append(total_eds)

        results.append({
            't': t, 
            'eds': total_eds
        })

    # Initial conditions for all objects: [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk]
    initial_conditions = [0.05, 0, 0.060, 0, 0.010, 0, 0.0, 0]
    time_total = 0.1
    t_span = (0, time_total)
    t_eval = np.linspace(0, time_total, 5000)

    # Solve the combined system with dense output
    sol_combined = solve_ivp(
        combined_equations, 
        t_span, 
        initial_conditions, 
        t_eval=t_eval, 
        method='RK45', 
        rtol=1e-6, 
        atol=1e-6,
        dense_output=True
    )

    # Получаем значения решения на точках `t_eval`
    solution_values = sol_combined.sol(t_eval)

    # Рассчитываем ЭДС на каждом шаге
    for i, t in enumerate(t_eval):
        z_m = solution_values[0, i]
        v_m = solution_values[1, i]
        save_eds_forces(z_m, v_m, t)

    # Plotting the results
    # plt.figure(figsize=(12, 10))
 
    # plt.plot(
    # t_eval, eds_forces, 
    #     color='red', 
    #     label=f"Сила ЭДС\nКоличество витков: {coil.turns_count}\n"
    #         f"Количество витков на слой: {coil.turns_per_layer}\n"
    #         f"Количество слоёв: {coil.layer_count}\n"
    #         f"Высота катушки: {coil.height}"
    # )
 


if __name__ == '__main__': 
    miews = sorted([f.split('.')[0] for f in os.listdir(csv_dir) if f.endswith('.csv')], key=extract_number)

    for miew in miews:
        print(f"Creating for μ={miew}")
        create_csv(int(miew))
        with open(f'csv/a2/{miew}.csv', mode='w', newline='') as csv_file:
            fieldnames = ['t', 'eds']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for data in results:
                writer.writerow(data) 