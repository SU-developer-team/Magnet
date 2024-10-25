from models_v2 import Magnet, shaker_force, G
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
    damping_coefficient = 0

    def equation_shaker(t, y):
        z, v = y  # Позиция и скорость 
        F_gravity = -magnet.mass * G    
        F_shaker = shaker_force(magnet, t)
        a = F_shaker / magnet.mass 
        return [v, a] 

    def equation_top_m(t, y):
        z, v = y  # Позиция и скорость 
        if z < z_top:
            F_top_magnetic = magnet.get_force(z - z_top)
        else:
            F_top_magnetic = 0 
        F_gravity = -magnet.mass * G    
        F_shaker = shaker_force(magnet, t)
        # F_top_magnetic -= F_gravity 
        F_top_magnetic = F_shaker
        a = F_top_magnetic / magnet.mass
        return [v, a] 
     
    def equation_bottom_m(t, y):
        z, v = y  # Позиция и скорость 
        if z < z_top:
            F_bottom_magnetic = magnet.get_force(z - z_top)
        else:
            F_bottom_magnetic = 0 
        F_gravity = -magnet.mass * G    
        F_shaker = shaker_force(magnet, t)
        F_bottom_magnetic -= F_gravity 
        F_bottom_magnetic = F_shaker
        a = F_bottom_magnetic / magnet.mass
        return [v, a]

    # Определение функции для интегратора solve_ivp
    def equations(t, y):
        z, v = y  # Позиция и скорость 

        # Сила от верхнего магнита (отталкивающая вниз)
        if z < z_top:
            F_top_magnetic = magnet.get_force(z - z_top)
        else:
            F_top_magnetic = 0

        # Сила от нижнего магнита (отталкивающая вверх)
        if z > z_bottom:
            F_bottom_magnetic = magnet.get_force(z - z_bottom)
        else:
            F_bottom_magnetic = 0

        # Сила тяжести
        F_gravity = magnet.mass * G

        # Сила шейкера
        F_shaker = shaker_force(magnet, t)

        # Сила демпфирования
        F_damping = -damping_coefficient * v

        # Общая сила как сумма всех сил
        F_total = -F_top_magnetic + F_bottom_magnetic - F_gravity + F_shaker + F_damping

        # Ускорение рассчитывается как F = m * a => a = F / m
        a = F_total / magnet.mass

        # Возвращаем скорость и ускорение
        return [v, a]

    # Задаем начальные условия [начальное положение, начальная скорость]
    initial_conditions = [0.015, 0]  # Начальная позиция и скорость
    initial_conditions_tm = [0.06, 0]  # Начальная позиция и скорость
    initial_conditions_bm = [0.01, 0]  # Начальная позиция и скорость
    initial_conditions_sk = [0.0, 0]  # Начальная позиция и скорость
    time_total = 1
    # Временные точки, для которых нужно рассчитать значения
    t_span = (0, time_total)  # Моделируем в течение time_total секунд
    t_eval = np.linspace(0, time_total, 5000)  # 1000 точек времени от 0 до time_total секунд

    # Решаем уравнение с использованием solve_ivp
    sol_total = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval, method='RK45')
    sol_top_m = solve_ivp(equation_top_m, t_span, initial_conditions_tm, t_eval=t_eval, method='RK45') 
    sol_bottom_m = solve_ivp(equation_bottom_m, t_span, initial_conditions_bm, t_eval=t_eval, method='RK45') 
    sol_shaker = solve_ivp(equation_shaker, t_span, initial_conditions_sk, t_eval=t_eval, method='RK45')

    # Печатаем результат решения
    if sol_total.success:
        print("Решение успешно найдено")
        print("sol_total", sol_total)
    else:
        print("Проблема с решением: ", sol_total.message)

    # Отображение результатов
    plt.figure(figsize=(10, 5))

    # График положения во времени 
    plt.plot(sol_total.t, sol_total.y[0], label='Магнит (z)', color='blue')
 
    # # Шейкер
    plt.plot(sol_shaker.t, sol_shaker.y[0], label='(Шейкер) Положение (z)', color='green')

    # # Верхний магнит
    plt.plot(sol_top_m.t, sol_top_m.y[0], label='(Верхний магнит) Положение (z)', color='purple')

    # # # Нижний магнит
    plt.plot(sol_bottom_m.t, sol_bottom_m.y[0], label='(Нижний магнит) Положение (z)', color='orange')
 
    plt.xlabel('Время (с)')
    plt.ylabel('Положение (м)')
    plt.legend()
    plt.grid()
    plt.show()

    # plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
