import math 


class Magnet:
    def __init__(self, mass, diameter, height):
        self.mass = mass
        self.diameter = diameter
        self.height = height
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
    
    def coordinate_params():
        pass
    

class Coil:
    def __init__(self, turns, radius, position=0):
        self.turns = turns
        self.radius = radius
        self.area = math.pi * radius**2
        self.position = position  # Положение катушки по оси z


def calculate_emf(magnet, coil, velocities, z_starts, z_ends, time_step=0.001):
    """
    Расчет ЭДС, индуцированной в катушке при прохождении магнита в нескольких фазах движения.
    velocities - список скоростей для каждой фазы
    z_starts - список начальных позиций для каждой фазы
    z_ends - список конечных позиций для каждой фазы
    """
    z_positions = []
    emfs = []
    times = []
    total_time = 0

    for v, z_start, z_end in zip(velocities, z_starts, z_ends):
        z = z_start
        t = total_time

        Phi_prev = None

        # Определяем направление движения
        direction = 1 if z_end > z_start else -1

        while (direction * z) <= (direction * z_end):
            # Сдвигаем положение относительно катушки
            z_rel = z - coil.position

            B = magnet.magnetic_induction(z_rel)
            Phi = coil.turns * B * coil.area

            if Phi_prev is not None:
                dPhi_dt = (Phi - Phi_prev) / time_step
                emf = -dPhi_dt
            else:
                emf = 0

            emfs.append(emf)
            times.append(t)
            z_positions.append(z)

            Phi_prev = Phi
            z += v * time_step
            t += time_step

        total_time = t  # Обновляем общее время для следующей фазы

    return times, emfs, z_positions

