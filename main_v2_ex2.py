from models.models_v2 import Magnet, shaker_force, G
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math


def main():
    # Создание объекта магнита
    magnet = Magnet(
        diameter=0.0195,  # Диаметр магнита (м) 19.5 мм
        mass=0.043,  # Масса магнита (кг) 43 г
        height=0.01  # Высота магнита (м) 10 мм
    )
    
    
    # Магнитные силы от верхнего и нижнего магнита
    z_top = 0.06
    z_bottom = 0.01

    # Коэффициент демпфирования
    damping_coefficient = 0.05

    # Define combined equations
    def combined_equations(t, y):
        z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk = y 

        F_gravity = magnet.mass * G

        # Main Magnet Forces
        F_shaker = shaker_force(magnet, t)
        # Magnet a
        a_sk = F_shaker / magnet.mass

        F_top_magnetic = magnet.get_force(abs(z_m - z_top)) if z_m < z_top else 0

        F_bottom_magnetic = magnet.get_force(abs(z_m - z_bottom)) if z_m >= z_bottom else 0
        print(f'Zm {z_m} | Zb {z_bottom} | dz {z_m - z_bottom}')       
        
        F_damping = -damping_coefficient * v_m

        F_total_magnet = -F_top_magnetic + F_bottom_magnetic - F_gravity + F_shaker + F_damping
        # top magnet a
        a_tm = F_shaker / magnet.mass
        # bottom magnet a
        a_bm = F_shaker / magnet.mass

        a_m = F_total_magnet / magnet.mass

        # print(f"Z m {z_m} | Zb {z_bm}")
        # Return combined derivatives
        return [v_m, a_m, v_tm, a_tm, v_bm, a_bm, v_sk, a_sk]

    # Initial conditions for all objects: [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk]
    initial_conditions = [0.015, 0, 0.06, 0, 0.01, 0, 0.0, 0]
    time_total = 20
    t_span = (0, time_total)
    t_eval = np.linspace(0, time_total, 1000)

    # Solve the combined system
    sol_combined = solve_ivp(
        combined_equations, 
        t_span, 
        initial_conditions, 
        t_eval=t_eval, 
        method='RK45', 
        rtol=1e-9, 
        atol=1e-9
    )
    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(sol_combined.t, sol_combined.y[0], label='Магнит (z)', color='blue')
    plt.plot(sol_combined.t, sol_combined.y[6], label='(Шейкер) Положение (z)', color='green')
    plt.plot(sol_combined.t, sol_combined.y[2], label='(Верхний магнит) Положение (z)', color='purple')
    plt.plot(sol_combined.t, sol_combined.y[4], label='(Нижний магнит) Положение (z)', color='orange')
    


    plt.xlabel('Время (с)')
    plt.ylabel('Положение (м)') 
    plt.legend()
    plt.grid()
    plt.show()

    d_z = [z - sol_combined.y[4][i] for i, z in enumerate(sol_combined.y[0])]

    f = [27 * np.exp(0.11 * z) for z in d_z]


    # plt.figure(figsize=(10, 5))
    # plt.plot(sol_combined.t, f, label='F(d_z)', color='red')
    # plt.xlabel('Расстояние d_z (м)')
    # plt.ylabel('Сила F (Н)')
    # plt.legend()
    # plt.grid()
    # plt.show()

if __name__ == '__main__':
    main()
