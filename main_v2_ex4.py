from models.models_v2 import Magnet, shaker_force, G, Coil, μ
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import csv

results = []
eds_forces = []


def main():
    # Создание объекта магнита
    magnet = Magnet(
        diameter=1.95,  # Диаметр магнита 19.5 (м) 
        mass=43,       # Масса магнита    43 (кг)
        height=10       # Высота магнита  10 (м)
    )

    # Магнитные силы от верхнего и нижнего магнита
    z_top = 6
    z_bottom = 1

    coil = Coil(
        turns=52,                # Уменьшенное количество витков
        thickness=1.025,       # Толщина катушки равна радиусу 1.025 мм
        radius=1.025,          # Радиус катушки 1.025 мм
        position=1,
        magnet=magnet,
        wire_diameter=0.5      # Диаметр провода 0.5 мм
    )




    # Коэффициент демпфирования
    damping_coefficient = 0.1
 

    # Список для хранения силы ЭДС
    eds_forces = []

    # Define combined equations with ЭДС
    def combined_equations(t, y):
        z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk = y 

        F_gravity = magnet.mass * G

        # Main Magnet Forces
        F_shaker = shaker_force(magnet, t)

        # Magnet acceleration
        a_sk = F_shaker / magnet.mass

        F_top_magnetic = magnet.get_force(abs(z_m - z_top)) if z_m < z_top else 0
        F_bottom_magnetic = magnet.get_force(abs(z_m - z_bottom)) if z_m > z_bottom else 0
        
        # Cd≈1.0 Ламинарный поток
        # Cd≈1.2  Турбулентный поток
        Cd = 1.2
        ro = 0.001225
        A = math.pi
        F_damping = 0.5*ro*v_m**2 * Cd * A

        # Общая сила с учётом ЭДС
        F_total_magnet = -F_top_magnetic + F_bottom_magnetic - F_gravity + F_shaker + F_damping
        
        # top magnet a
        a_tm = F_shaker / magnet.mass
        # bottom magnet a
        a_bm = F_shaker / magnet.mass

        a_m = F_total_magnet / magnet.mass

        # Return combined derivatives
        return [v_m, a_m, v_tm, a_tm, v_bm, a_bm, v_sk, a_sk]

    # Обратный вызов для записи значения ЭДС в моменты `t_eval`
    def save_eds_forces(z_m, v_m, t):
        eds = coil.get_eds(z_m, v_m, t)
        # print('EDS:', eds)
        eds_forces.append(eds)

        results.append({
            't': t,
            'z_m': z_m,
            'v_m': v_m,
            'eds': eds
        })


        

    # Initial conditions for all objects: [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk]
    initial_conditions = [0.025, 0, 0.06, 0, 0.01, 0, 0.0, 0]
    time_total = 10
    t_span = (0, time_total)
    t_eval = np.linspace(0, time_total, 1000)

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
    plt.plot(sol_combined.t, [coil.position for _ in sol_combined.t], label='Катушка', color='red')
    plt.xlabel('Время (с)')
    plt.ylabel('Положение (м)') 
    plt.legend()
    plt.grid()

    # ЭДС
    plt.subplot(3, 1, 2)
    plt.plot(t_eval, eds_forces, label='Сила ЭДС', color='red')
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
    print('LEN eds_forces:', len(eds_forces))
    print('sol_combined.t LEN:', len(sol_combined.t))







if __name__ == '__main__':
    main()
    with open(f'{μ}.csv', mode='w', newline='') as csv_file:
        fieldnames = ['t', 'z_m', 'v_m', 'eds']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for data in results:
            writer.writerow(data)


