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

    # Коэффициент демпфирования
    damping_coefficient = 0.1

    # Определение функции для интегратора solve_ivp
    def equations(t, y):
        z, v = y  # Позиция и скорость

        # Магнитные силы от верхнего и нижнего магнита
        z_top = 0.10
        z_bottom = -0.10

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
        F_gravity = -magnet.mass * G

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
    initial_conditions = [0, 0]  # Начальная позиция и скорость
    time_total = 5
    # Временные точки, для которых нужно рассчитать значения
    t_span = (0, time_total)  # Моделируем в течение time_total секунд
    t_eval = np.linspace(0, time_total, 1000)  # 1000 точек времени от 0 до time_total секунд

    # Решаем уравнение с использованием solve_ivp
    sol = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval, method='RK45')

    # Печатаем результат решения
    if sol.success:
        print("Решение успешно найдено")
    else:
        print("Проблема с решением: ", sol.message)

    # Отображение результатов
    plt.figure(figsize=(10, 5))

    # График положения во времени
    plt.subplot(2, 1, 1)
    plt.plot(sol.t, sol.y[0], label='Положение (z)', color='blue')
    plt.xlabel('Время (с)')
    plt.ylabel('Положение (м)')
    plt.legend()
    plt.grid()

    # График скорости во времени
    plt.subplot(2, 1, 2)
    plt.plot(sol.t, sol.y[1], label='Скорость (v)', color='red')
    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
