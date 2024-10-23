import math

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

    def motion(self, F_external_func, z0, v0, t_total, time_step=0.001, z_min=-0.05, z_max=0.05):
        """
        Расчет ускорения, скорости и позиции магнита во времени под действием внешней силы и силы тяжести.
        Теперь магнит ограничен границами z_min и z_max.
        
        F_external_func - функция, возвращающая внешнюю силу в зависимости от времени, положения и скорости
        z0 - начальное положение (м)
        v0 - начальная скорость (м/с)
        t_total - общее время моделирования (с)
        time_step - шаг по времени (с)
        z_min - нижняя граница по оси z
        z_max - верхняя граница по оси z
        """
        times = []
        positions = []
        velocities = []
        accelerations = []

        t = 0
        z = z0
        v = v0
        m = self.mass
        g = 9.81  # Ускорение свободного падения (м/с^2)

        while t <= t_total:
            # Получаем внешнюю силу в текущий момент времени
            F_external = F_external_func(t, z, v)

            # Расчет ускорения
            a = (F_external - m * g) / m

            # Обновление скорости и позиции (метод Эйлера)
            v += a * time_step
            z += v * time_step

            # Проверка на границы и отражение
            if z >= z_max:
                z = z_max
                v = -v  # Инвертируем скорость
            elif z <= z_min:
                z = z_min
                v = -v  # Инвертируем скорость

            # Сохранение данных
            times.append(t)
            positions.append(z)
            velocities.append(v)
            accelerations.append(a)

            # Обновление времени
            t += time_step

        return times, positions, velocities, accelerations


class Coil:
    def __init__(self, turns, radius, position=0):
        self.turns = turns
        self.radius = radius
        self.area = math.pi * radius**2
        self.position = position  # Положение катушки по оси z

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

        # Сдвигаем положение относительно катушки
        z_rel = z - coil.position

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
