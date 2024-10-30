from models.models_v2 import Magnet, shaker_force, G, Coil, μ
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import csv
import logging
from datetime import datetime 


# Set up logger
logger = logging.getLogger('magnet_simulation')
logger.setLevel(logging.DEBUG)
timestamp = datetime.now().strftime('%d.%m.%Y-%H-%M-%S')

# Create file handler which logs even debug messages
fh = logging.FileHandler(f'logs/{timestamp}.log')
fh.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

results = []
eds_forces = []


def main():
    # Создание объекта магнита
    magnet = Magnet(
        diameter=0.0195,  # Диаметр магнита 19.5 (мм) 
        mass=0.043,       # Масса магнита    43 (г)
        height=0.01       # Высота магнита  10 (мм)
    )

    # Позиция магнитов
    z_top = 0.06
    z_bottom = 0.01
    

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
        F_shaker = shaker_force(magnet, t)
        a_sk = F_shaker / magnet.mass  # Ускорение шейкера

        F_top_magnetic = magnet.get_force(abs(z_m+(magnet.height/2) - z_top), F_shaker) 
        F_bottom_magnetic = magnet.get_force(abs(z_m+(magnet.height/2) - z_bottom), F_shaker)

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

        # Логирование (опционально)
        logger.info(f"F total: {F_total_magnet}")
        logger.info(f"F_top_magnetic: {-F_top_magnetic}")
        logger.info(f"F_bottom_magnetic: {F_bottom_magnetic}")
        logger.info(f"F_gravity: {-F_gravity}")
        logger.info(f"F_shaker: {F_shaker}")
        logger.info(f"F_damping: {F_damping}")
        logger.info(f"Z {z_m}")
        logger.info(f"Time {t}\n---------------------------------------------")

        # Вычисление ускорений
        a_m = F_total_magnet / magnet.mass  # Ускорение главного магнита
        a_tm = F_shaker / magnet.mass   # Ускорение верхнего магнита
        a_bm = F_shaker / magnet.mass  # Ускорение нижнего магнита

        # Возврат производных переменных состояния
        return [v_m, a_m, v_tm, a_tm, v_bm, a_bm, v_sk, a_sk]


    # Обратный вызов для записи значения ЭДС в моменты `t_eval`
    def save_eds_forces(z_m, v_m, t):
        eds_per_turn, total_eds = coil.get_eds(z_m, v_m, t)
        # print('EDS:', eds)
        eds_forces.append(total_eds)

        results.append({
            't': t,
            'z_m': z_m,
            'v_m': v_m,
            'eds': total_eds
        })


        

    # Initial conditions for all objects: [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk]
    initial_conditions = [0.05, 0, 0.060, 0, 0.010, 0, 0.0, 0]
    time_total = 1
    t_span = (0, time_total)
    t_eval = np.linspace(0, time_total, 500)

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
    plt.figure(figsize=(12, 10))

    # Положение магнита и шейкера
    plt.subplot(3, 1, 1)
    plt.plot(sol_combined.t, sol_combined.y[0], label='Магнит (z)', color='blue')
    plt.plot(sol_combined.t, sol_combined.y[6], label='(Шейкер) Положение (z)', color='green')
    plt.plot(sol_combined.t, sol_combined.y[2], label='(Верхний магнит) Положение (z)', color='purple')
    plt.plot(sol_combined.t, sol_combined.y[4], label='(Нижний магнит) Положение (z)', color='orange')
    plt.plot(sol_combined.t, [coil.position for _ in sol_combined.t], label='Катушка нижний граница', color='red')
    plt.plot(sol_combined.t, [(coil.position + coil.height) for _ in sol_combined.t], label='Катушка верхний граница', color='red')
    plt.xlabel('Время (с)')
    plt.ylabel('Положение (м)') 
    plt.legend()
    plt.grid()

    # ЭДС
    t_csv = []
    v_csv = []
    with open('exp_csv/10.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            t_csv.append(float(row[0]))
            v_csv.append(float(row[1]))
 
    t_csv_start = t_csv[0]
    t_csv = [t - t_csv_start for t in t_csv]



    plt.subplot(3, 1, 2)
    # plt.plot(t_eval, eds_forces, label='Сила ЭДС', color='red', label=f"Количество витков: {coil.turns_count}\nКоличество витков на слой: {coil.turns_per_layer}\nКоличество слойев: {coil.layer_count}\nВысота катушки: {coil.height}")
    plt.plot(
    t_eval, eds_forces, 
        color='red', 
        label=f"Сила ЭДС\nКоличество витков: {coil.turns_count}\n"
            f"Количество витков на слой: {coil.turns_per_layer}\n"
            f"Количество слоёв: {coil.layer_count}\n"
            f"Высота катушки: {coil.height}"
    )

    # Построение второго графика с данными из CSV
    plt.plot(
        t_csv, v_csv, 
        color='blue', 
        label='Данные из CSV'
    )


    plt.xlabel('Время (с)')
    plt.ylabel('ЭДС (V)')
    plt.legend()
    plt.grid()

    # Скорость магнита
    plt.subplot(3, 1, 3)
    plt.plot(sol_combined.t, sol_combined.y[1], label='Скорость магнита (v)', color='cyan')
    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()  


if __name__ == '__main__':
    main()
    with open(f'csv/{μ}.csv', mode='w', newline='') as csv_file:
        fieldnames = ['t', 'z_m', 'v_m', 'eds']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for data in results:
            writer.writerow(data)

    logger.info('----------------END----------------')