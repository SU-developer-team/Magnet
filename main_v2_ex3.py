from models.models_v2 import Magnet, shaker_force, G, Coil
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

def main():
    # Создание объекта магнита
    magnet = Magnet(
        diameter=0.0195,  # Диаметр магнита (м) 19.5 мм
        mass=0.043,       # Масса магнита (кг) 43 г
        height=0.01       # Высота магнита (м) 10 мм
    )

    # Магнитные силы от верхнего и нижнего магнита
    z_top = 0.06
    z_bottom = 0.01

    coil = Coil(
        turns=200,
        radius=0.0002,
        position=0.02
    )

    # Коэффициент демпфирования
    damping_coefficient = 0.1

    # Коэффициент ЭДС
    eds_coefficient = 0.20

    # Список для хранения силы ЭДС
    eds_forces = []

    # Define combined equations with ЭДС
    def combined_equations(t, y):
        z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk = y 

        F_gravity = magnet.mass * G

        # Main Magnet Forces
        F_shaker = shaker_force(magnet, t)
        
        # ЭДС сила
        # F_eds = -eds_coefficient * v_m

        # Magnet acceleration
        a_sk = F_shaker / magnet.mass

        F_top_magnetic = magnet.get_force(abs(z_m - z_top)) if z_m < z_top else 0
        F_bottom_magnetic = magnet.get_force(abs(z_m - z_bottom)) if z_m >= z_bottom else 0
        
        F_damping = -damping_coefficient * v_m

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
        eds_forces.append(eds)

    # Initial conditions for all objects: [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk]
    initial_conditions = [0.025, 0, 0.06, 0, 0.01, 0, 0.0, 0]
    time_total = 2
    t_span = (0, time_total)
    t_eval = np.linspace(0, time_total, 100000)

    # Solve the combined system with dense output
    sol_combined = solve_ivp(
        combined_equations, 
        t_span, 
        initial_conditions, 
        t_eval=t_eval, 
        method='RK45', 
        rtol=1e-9, 
        atol=1e-9,
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
    plt.plot(sol_combined.t, [0.025 for _ in sol_combined.t], label='Катушка', color='red')
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
