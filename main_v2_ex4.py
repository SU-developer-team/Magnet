from models_v2 import Magnet, shaker_force, G
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Константы и параметры системы
damping_coefficient = 0.05
z_top = 0.06
z_bottom = 0.01

# Создаем объект магнита
magnet = Magnet(
    diameter=0.0195,  # Диаметр магнита (м)
    mass=0.043,       # Масса магнита (кг)
    height=0.01       # Высота магнита (м)
)

# Числовые функции для магнитных сил
def get_force_top(z_m_val):
    """Сила верхнего магнита"""
    return magnet.get_force(abs(z_m_val - z_top))

def get_force_bottom(z_m_val, a_m):
    """Сила нижнего магнита с учетом условия для a_m * mass"""
    if z_m_val > z_bottom:
        return magnet.get_force(abs(z_m_val - z_bottom))
    else:
        return magnet.get_force(abs(z_bottom - z_m_val)) + a_m * magnet.mass

# Главная функция интеграции
def main():
    # Начальные условия: [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk]
    initial_conditions = [0.015, 0, 0.06, 0, 0.01, 0, 0.0, 0]
    time_total = 2
    t_span = (0, time_total)
    t_eval = np.linspace(0, time_total, 1000)

    # Определение системы уравнений
    def combined_equations(t, y):
        z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk = y

        # Внешняя сила от шейкера
        F_shaker = shaker_force(magnet, t)

        # Вычисляем магнитные силы
        F_top_magnetic = get_force_top(z_m) if z_m < z_top else magnet.mass * G

        # Расчет F_total_magnet с a_m
        F_damping = -damping_coefficient * v_m
        F_total_magnet_preliminary = -F_top_magnetic - magnet.mass * G + F_shaker + F_damping + get_force_bottom(z_m)
        a_m = F_total_magnet_preliminary / magnet.mass

        # Корректировка F_bottom_magnetic с использованием a_m
        F_bottom_magnetic = get_force_bottom(z_m, a_m)

        # Общая сила, действующая на магнит
        F_total_magnet = F_total_magnet_preliminary + F_bottom_magnetic

        # Ускорения
        a_m_final = F_total_magnet / magnet.mass
        a_tm = F_shaker / magnet.mass
        a_bm = F_shaker / magnet.mass
        a_sk = F_shaker / magnet.mass

        # Производные
        dz_m_dt = v_m
        dv_m_dt = a_m_final
        dz_tm_dt = v_tm
        dv_tm_dt = a_tm
        dz_bm_dt = v_bm
        dv_bm_dt = a_bm
        dz_sk_dt = v_sk
        dv_sk_dt = a_sk

        return [dz_m_dt, dv_m_dt, dz_tm_dt, dv_tm_dt, dz_bm_dt, dv_bm_dt, dz_sk_dt, dv_sk_dt]

    # Решение системы уравнений
    sol_combined = solve_ivp(combined_equations, t_span, initial_conditions, t_eval=t_eval, method='RK45')

    # Построение графиков
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

if __name__ == '__main__':
    main()
