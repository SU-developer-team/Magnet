import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import logging
from datetime import datetime
from models import Magnet, Shaker, Coil

# Допустим, у вас есть отдельные классы в models.py
# from models import Magnet, Shaker, Coil 
# ===========================================================
# Функции для расчёта разных сил и т.д.

def get_magnet_position(shaker, t):
    return shaker.X0 * np.sin(shaker.W * t)

def calculate_f_damping(v_m, magnet):
    """
    Расчет силы демпфирования (сопротивления воздуха).
    """
    Cd = 1.2  # Коэффициент лобового сопротивления (для турбулентного потока)
    ro = 1.225  # Плотность воздуха (кг/м^3)
    A = math.pi * magnet.diameter ** 2 * 0.25  # Площадь поперечного сечения магнита
    F_damping = 0.1 * ro * (v_m**2) * Cd * np.sign(v_m)
    return F_damping

def calculate_f_air(z_m, v_m, magnet, z_top, z_bottom):
    """
    Расчет силы вязкого трения воздуха в зазоре между магнитом и цилиндром.
    """
    D_outer = 0.0205  # Диаметр внутренней стенки цилиндра (м)
    D_inner = magnet.diameter  # Диаметр магнита (м)
    gap = (D_outer - D_inner) / 2  # Радиальный зазор (м)

    if gap <= 0:
        raise ValueError("Диаметр магнита должен быть меньше диаметра цилиндра.")

    mu_air = 1.81e-5  # Динамическая вязкость воздуха (Па·с)

    # Простейшая формула для кольцевого зазора:
    F_viscous = -6 * np.pi * mu_air * magnet.height * v_m / gap
    return F_viscous

# ===========================================================
# 1) Система ОДУ
def combined_equations(t, y, magnet, shaker, z_top, z_bottom, coil, resistance):
    """
    Система дифференциальных уравнений для расчета движения магнита и тока в катушке.
    y = [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk, i].
    """
    (z_m, v_m,
     z_tm, v_tm,
     z_bm, v_bm,
     z_sk, v_sk,
     i_current) = y

    F_gravity = magnet.mass * shaker.G
    # Сила от шейкера (упрощённый пример)
    F_shaker = shaker.get_force(magnet, t)

    # Допустим, ускорение шейкера:
    a_sk = F_shaker / magnet.mass

    # Сила от "верхнего магнита":
    F_top_magnetic = magnet.get_force(
        abs(z_m - (magnet.height / 2) - z_top) + get_magnet_position(shaker, t)
    )
    # Сила от "нижнего магнита":
    F_bottom_magnetic = magnet.get_force(
        abs(z_m - (magnet.height / 2) - z_bottom) + get_magnet_position(shaker, t)
    )

    # Сила лобового сопротивления
    F_damping = calculate_f_damping(v_m, magnet)
    # Сила вязкого трения воздуха
    F_viscous = calculate_f_air(z_m, v_m, magnet, z_top, z_bottom)

    # Итоговая сила на магнит
    F_total_magnet = (- F_top_magnetic
                      + F_bottom_magnetic
                      - F_gravity
                      - F_damping
                      + F_viscous
                      )

    a_m = F_total_magnet / magnet.mass  # ускорение магнита
    a_tm = F_shaker / magnet.mass       # ускорение "верхнего магнита"
    a_bm = F_shaker / magnet.mass       # ускорение "нижнего магнита"

    # ЭДС
    eds_per_turn, total_eds = coil.get_total_emf(shaker, z_m, v_m, t, a_m)
    inductance = coil.calculate_inductance()

    # dI/dt
    di_dt = (total_eds - resistance * i_current) / inductance

    # Возвращаем производные:
    return [
        v_m,             # dz_m/dt
        a_m,             # dv_m/dt
        v_tm,            # dz_tm/dt
        a_tm,            # dv_tm/dt
        v_bm,            # dz_bm/dt
        a_bm,            # dv_bm/dt
        v_sk,            # dz_sk/dt
        a_sk,            # dv_sk/dt
        di_dt            # dI/dt
    ]

# ===========================================================
# 2) Функции событий (event)

def event_bottom(t, y, magnet, shaker, z_bottom, z_top, coil, resistance):
    """
    Срабатывает, когда z_m == z_bottom (приход 'сверху вниз').
    """
    z_m = y[0]
    return z_m - z_bottom

event_bottom.terminal = True    # Останавливаем интеграцию
event_bottom.direction = -1     # Условие ловится, когда (z_m - z_bottom) переходит через 0 в направлении убывания

def event_top(t, y, magnet, shaker, z_bottom, z_top, coil, resistance):
    """
    Срабатывает, когда z_m == z_top (приход 'снизу вверх').
    """
    z_m = y[0]
    return z_m - z_top

event_top.terminal = True
event_top.direction = 1

# ===========================================================
def main():
    # --- Логгер (опционально) ---
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger('magnet_simulation')

    # --- Параметры системы ---
    z_top = 0.09
    z_bottom = 0.01
    G = 9.8
    X0 = 0.001
    freq = 50           # Частота колебаний
    time_total = 2.0
    magnet_start_z = 0.0425

    # Создаём объекты
    magnet = Magnet(diameter=0.0195, mass=0.021, height=0.01)
    shaker = Shaker(G=G, miew=freq, X0=X0)
    coil = Coil(
        turns_count=208,
        thickness=0.01025,
        radius=0.01025,
        position=0.015,
        magnet=magnet,
        wire_diameter=0.000961538462,
        layer_count=4,
    )
    resistance = 0.1

    # --- Начальные условия ---
    # y = [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk, i]
    # Допустим, у "верхнего магнита" и "нижнего магнита" положения = z_top, z_bottom,
    # у шейкера = 0, скорости все = 0, ток i=0
    y0 = [
        magnet_start_z,  # z_m
        0.0,             # v_m
        z_top,           # z_tm
        0.0,             # v_tm
        z_bottom,        # z_bm
        0.0,             # v_bm
        0.0,             # z_sk
        0.0,             # v_sk
        0.0              # i
    ]

    t_start = 0.0

    # --- Массивы для "склейки" решения ---
    T_global = []
    Y_global = []

    # Текущее время и текущее состояние
    current_t = t_start
    current_y = np.array(y0, dtype=float)

    # Пока не пройдём всё время: 

        # Запускаем solve_ivp на [current_t, time_total]
    sol = solve_ivp(
        fun=combined_equations,
        t_span=(0, time_total),
        y0=current_y,
        method='RK45',
        args=(magnet, shaker, z_bottom, z_top, coil, resistance),
        events=[event_bottom, event_top],
        rtol=1e-6,
        atol=1e-6,
        max_step=1e-3,   # чтобы точнее ловить момент события
        dense_output=True
    )

    # "Склеиваем" решение
    if len(T_global) == 0:
        T_global = sol.t
        Y_global = sol.y
    else:
        # Избавляемся от дублирования последней точки,
        # т.к. она совпадает с первым шагом следующего куска
        T_global = np.concatenate((T_global, sol.t[1:]))
        Y_global = np.concatenate((Y_global, sol.y[:, 1:]), axis=1)

    # Проверяем, было ли событие
    # events: sol.t_events[0] = моменты касания bottom
    #    sol.t_events[1] = моменты касания top
    bottom_times = sol.t_events[0]
    top_times = sol.t_events[1]

    if (len(bottom_times) == 0) and (len(top_times) == 0):
        # Ни одно событие не сработало -> дошли до time_total
        # Выходим из цикла
        pass
    else:
        # Событие есть -> интеграция остановилась в момент события
        # Это конечная точка sol (последняя)
        collision_t = sol.t[-1]
        collision_y = sol.y[:, -1].copy()

        # collision_y = [z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk, i]
        # Нужно "отразить" скорость магнита v_m = y[1]
        # Если хотим упругий удар: v_m -> -v_m
        # или с коэффициентом восстановления e < 1
        e = 1.0  # абсолютно упругое
        collision_v_m = collision_y[1]
        collision_y[1] = - e * collision_v_m

        # Готовим следующие начальные условия
        current_t = collision_t
        current_y = collision_y

        # Чтобы не застрять на событии, иногда добавляют маленький сдвиг
        # current_t += 1e-9
        # Но часто solve_ivp сам двигается дальше в следующем шаге.

    # ---- Всё, у нас есть массивы T_global, Y_global со "склеенным" решением ----

    # Вычислим ЭДС постфактум (если нужно)
    # Y_global[0,:] - это z_m по временам,
    # Y_global[1,:] - это v_m,
    # ...
    z_m_values = Y_global[0, :]
    v_m_values = Y_global[1, :]
    i_values = Y_global[8, :]  # Ток
    T_global = np.array(T_global)

    # Рассчитаем ускорение магнита (через числ. производную)
    a_m_values = np.gradient(v_m_values, T_global)

    # Теперь посчитаем ЭДС на каждом шаге
    total_eds_values = []
    for i in range(len(T_global)):
        t = T_global[i]
        z_m = z_m_values[i]
        v_m = v_m_values[i]
        a_m = a_m_values[i]
        _, total_eds = coil.get_total_emf(shaker, z_m, v_m, t, a_m)
        total_eds_values.append(total_eds)
    total_eds_values = np.array(total_eds_values)

    # ЭДС самоиндукции
    inductance = coil.calculate_inductance()
    emf_self_induction = -inductance * np.gradient(i_values, T_global)

    # Итоговая ЭДС
    total_emf_with_self_induction = total_eds_values + emf_self_induction

    sum_energy = sum([emf**2 for emf in total_emf_with_self_induction])
    sum_self_energy = sum([emf**2 for emf in emf_self_induction])
    # --- Построим графики ---
    plt.figure(figsize=(10,8))
 
    plt.subplot(1, 1, 1)
    plt.plot(
        sol.t,
        total_eds_values,
        label=f'Внешняя ЭДС (total_eds)\nW total = {round(sum_energy, 4)}',
        color='blue',
        linewidth=0.5
    )
    plt.plot(
        sol.t,
        emf_self_induction,
        label=f'ЭДС самоиндукции (self_emf)\W self = {sum_self_energy}\nW = {round(sum_energy - sum_self_energy, 4)}',
        color='red',
        linewidth=0.5
    )
    plt.xlabel('Время (с)')
    plt.ylabel('ЭДС (В)')
    plt.legend()
    plt.grid()

 
    plt.savefig(f'saved_h_{z_top}_{freq}.png')
    plt.show() 

    logger.info('----------------END----------------')

if __name__ == '__main__':
    main()
