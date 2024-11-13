from models import Magnet, Shaker, Coil
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import csv 
from datetime import datetime 
import os
import re
import pandas as pd

μ = 5     # Частота колебаний

for μ in range(5, 105, 5):
    print(f"===== Расчёт для μ={μ} =====")
    results = []
    eds_forces = [] 
    csv_dir = "exp_csv/a2"

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
        z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk, i = y  # Добавили i в список переменных состояния

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

    def get_s(z_top):
        # Создание объекта магнита
        magnet = Magnet(
            diameter=0.0195,  # Диаметр магнита 19.5 мм
            mass=0.043,       # Масса магнита 43 г
            height=0.01,      # Высота магнита 10 мм
        )

        # Позиция магнитов
        z_bottom = 0.01
        G = 9.8  # Ускорение свободного падения (м/с^2)
        X0 = 0.001  # Амплитуда колебаний
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
        initial_conditions = [magnet_start_z, 0, z_top, 0, z_bottom, 0, 0.0, 0, 0]  # Добавили начальный ток 0
        t_span = (0, time_total)
        t_eval = np.linspace(0, time_total, 1000)

        # Решение системы уравнений
        sol_combined = solve_ivp(
            combined_equations,
            t_span,
            initial_conditions,
            args=(magnet, shaker, z_top, z_bottom, coil, resistance),  # Добавили coil и resistance
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-6,
        )

        # Получаем значения решения на точках `t_eval`
        solution_values = sol_combined.y
        a_values = np.gradient(solution_values[1], sol_combined.t)  # Используем градиент для численного дифференцирования

        # Рассчитываем ЭДС на каждом шаге
        total_eds_values = []
        for i, t in enumerate(t_eval):
            z_m = solution_values[0, i]
            v_m = solution_values[1, i]
            a_m = a_values[i]
            i_current = solution_values[8, i]

            # Расчет ЭДС
            eds_per_turn, total_eds = coil.get_total_emf(shaker, z_m, v_m, t, a_m)
            total_eds_values.append(total_eds)

        total_eds_values = np.array(total_eds_values)
        i_values = solution_values[8]  # Ток в катушке

        # Рассчитываем ЭДС самоиндукции на каждом шаге
        inductance = coil.calculate_inductance()
        di_dt = np.gradient(i_values, sol_combined.t)
        emf_self_induction = -inductance * di_dt

        # Итоговая ЭДС с учетом самоиндукции
        total_emf_with_self_induction = total_eds_values + emf_self_induction

        # Вычисляем среднеквадратическое значение ЭДС
        s = math.sqrt(np.mean(total_emf_with_self_induction ** 2))

        return s

    if __name__ == '__main__':  
        top_magnet_positions = np.arange(0.05, 0.14, 0.005)
        z_values = []
        s_values = []

        for z in top_magnet_positions:
            z = round(z, 3)
            s = get_s(z)
            s_values.append(s)
            z_values.append(z)
            print(f"Расчёт среднеквадратического значения ЭДС для высоты верхнего магнита z={z} окончен! s={s}")

        data = pd.DataFrame({
            'z': z_values,
            's': s_values
        })
        data.to_csv(f'z_h/csv/s_z__{μ}.csv', index=False)
        print(f"Данные успешно записаны в z_h__{μ}.csv")
        
        plt.figure(figsize=(20, 6))
        plt.plot(z_values, s_values, marker='o', linestyle='-', color='b', label='s(z)')
        plt.title('Зависимость s от z')
        plt.xlabel('z (Высота верхнего магнита)')
        plt.ylabel('s (Среднеквадратическое значение ЭДС)')
        plt.xticks(np.arange(0.05, 0.21, 0.01))
        plt.legend()
        plt.grid(True)
        plt.savefig(f'z_h/png/s_z__{μ}.png')
        # plt.show()                       # Показать график на экране
