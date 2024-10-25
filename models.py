import math
import numpy as np
from scipy.integrate import solve_ivp

class Magnet:
    def __init__(self, mass, diameter, height):
        self.mass = mass  # Масса магнита (кг)
        self.diameter = diameter  # Диаметр магнита (м)
        self.height = height  # Высота магнита (м)
        self.radius = diameter / 2 

    def magnetic_induction(self, z):
        """
        Расчет магнитной индукции в точке z по оси магнита.
        z - расстояние от центра магнита вдоль оси.
        """
        Br = 1.2  # Остаточная магнитная индукция магнита (Тл)
        R = self.radius
        L = self.height / 2

        def Bz(z, R, L):
            return (Br / 2) * (
                (z + L) / math.sqrt(R**2 + (z + L)**2) -
                (z - L) / math.sqrt(R**2 + (z - L)**2)
            )

        return Bz(z, R, L)

    def motion(self, F_external_func, z0, v0, t_total, time_step=0.001, z_min=-0.1, z_max=0.1):
        m = self.mass
        g = 9.81
        
        # Определяем систему уравнений
        def equations(t, y):
            z, v = y
            
            # Рассчитываем внешнюю силу (магнитные силы)
            F_external = F_external_func(t, z, v)
            
            # Общая сила с учётом силы тяжести и магнитной силы
            a = (F_external - m * g) / m  # Ускорение
            return [v, a]  # dz/dt = v, dv/dt = a

        # Временные точки для расчета
        t_eval = np.arange(0, t_total + time_step, time_step)

        # Начальные условия [z0, v0]
        initial_conditions = [z0, v0]

        # Решаем уравнение с использованием solve_ivp (метод Рунге-Кутты)
        solution = solve_ivp(equations, [0, t_total], initial_conditions, t_eval=t_eval, method='RK45')
        print(solution)
        # Ограничение значений z по границам z_min и z_max
        z = np.clip(solution.y[0], z_min, z_max)
        v = solution.y[1]
        slowing_radius = 0.03  # Радиус замедления при приближении к границам
        
        # Пересчитываем скорости при отражении
        for i in range(1, len(z)):
            # Замедление при приближении к границам
            if z_max - z[i] < slowing_radius:
                slowing_factor = (z_max - z[i]) / slowing_radius
                v[i] *= slowing_factor
            elif z[i] - z_min < slowing_radius:
                slowing_factor = (z[i] - z_min) / slowing_radius
                v[i] *= slowing_factor

            # Обновляем позицию на основе текущей скорости
            z[i] = z[i-1] + v[i-1] * time_step

            # Отражение от границ
            if z[i] == z_max or z[i] == z_min:
                v[i] = -v[i]  # Демпфируем скорость при отражении
                z[i] = np.clip(z[i], z_min, z_max)  # Убедимся, что z остается в пределах границ


        # Получаем времена, позиции, скорости и ускорения
        times = solution.t
        positions = z
        velocities = v
        accelerations = (F_external_func(times, positions, velocities) - m * g) / m

        return times, positions, velocities, accelerations


class Coil:
    def __init__(self, turns, radius, position=0, shaker_amplitude=0.01, shaker_frequency=2 * np.pi, shaker_phase=0):
        self.turns = turns
        self.radius = radius
        self.area = math.pi * radius**2
        self.position = position  # Положение катушки по оси z
        # Параметры для учета движения катушки
        self.shaker_amplitude = shaker_amplitude
        self.shaker_frequency = shaker_frequency
        self.shaker_phase = shaker_phase

    def coil_position(self, t):
        """
        Возвращает положение катушки в момент времени t с учетом колебаний шейкера.
        """
        return self.position + self.shaker_amplitude * np.sin(self.shaker_frequency * t + self.shaker_phase)

def calculate_emf(magnet, coil, times, positions):
    """
    Расчет ЭДС, индуцированной в катушке при движении магнита.
    times - список времен
    positions - список позиций магнита
    """
    emfs = []
    Phi_prev = None

    for i in range(len(times)):
        t = times[i]
        z = positions[i]

        # Положение катушки в момент времени t с учётом движения шейкера
        z_coil = coil.coil_position(t)

        # Относительное положение магнита относительно катушки
        z_rel = z - z_coil

        B = magnet.magnetic_induction(z_rel)
        Phi = coil.turns * B * coil.area

        if Phi_prev is not None:
            dt = times[i] - times[i - 1]
            dPhi_dt = (Phi - Phi_prev) / dt
            emf = -dPhi_dt
        else:
            emf = 0

        emfs.append(emf)
        Phi_prev = Phi

    return emfs
