import math
import numpy as np

G = 9800 # ускорение свободного падения (м/с^2) г. Алматы
X0 = 1 # амплитуда колебаний
μ  = 1 # частота колебаний
W = 2 * math.pi * μ 

# a = 0.0005
a = 78.614069
# b = 1.92
# a = 27
b= 0.11

class Magnet:
    def __init__(self, diameter, height, mass):
        self.radius = diameter / 2
        self.height = height
        self.mass = mass

    def get_force(self, z):
        """
        Расчет силы магнита в точке z по оси магнита.
        """
        k = 10
        #return k*z
        # return a * np.exp(-b * z)
        # print( a / z ** 2)
        return a / z
        # return a / (self.radius**2 + z**2)**1.5
    
    # def 
    
    def magnetic_induction(self, z):
        """
        Расчет магнитной индукции в точке z по оси магнита.
        z - расстояние от центра магнита вдоль оси.
        """
        Br = 0.1  # Остаточная магнитная индукция магнита (Тл)
        R = self.radius
        L = self.height / 2

        def Bz(z, R, L):
            return (Br / 2) * (
                (z + L) / math.sqrt(R**2 + (z + L)**2) -
                (z - L) / math.sqrt(R**2 + (z - L)**2)
            )

        return Bz(z, R, L)


class Coil:
    def __init__(self, turns, thickness, radius, position, magnet, wire_diameter=0.0005):
        self.turns = turns
        self.radius = radius
        self.area = math.pi * radius**2
        self.position = position  # Положение катушки по оси z 
        self.magnet = magnet
        self.thickness = thickness  # Толщина катушки (м)   
        self.wire_diameter = wire_diameter  # Диаметр провода (м)

        # Вычисление числа слоёв
        self.num_layers = max(1, int(self.thickness / self.wire_diameter))
        # Число витков на слой
        self.turns_per_layer = self.turns / self.num_layers
        # Вычисление высоты катушки
        self.height = self.turns * self.wire_diameter

    def coil_position(self, t):
        """
        Возвращает положение катушки в момент времени t с учетом колебаний шейкера.
        """
        return self.position + X0 * np.sin(W * t)

    # def get_eds(self, magnet_position, magnet_velocity, t):
    #     """
    #     Рассчитывает ЭДС, индуцированную в катушке, в зависимости от положения и скорости магнита.
    #     """
    #     # Расстояние между магнитом и катушкой
    #     distance = abs(magnet_position - self.coil_position(t))

    #     # Простейшая модель магнитного поля, убывающего с расстоянием, например, B ~ 1/distance^2
    #     B = 1 / distance**2 if distance > 0 else 0  # Избегаем деления на ноль

    #     # Производная магнитного потока по времени для ЭДС
    #     dPhi_dt = -self.turns * self.area * B * magnet_velocity

    #     return dPhi_dt

    def get_eds_v1(self, magnet_position, magnet_velocity, t):
        # Определяем расстояние от магнита до катушки
        distance = magnet_position - self.coil_position(t)

        # Используем точный расчет магнитной индукции магнита
        B = self.magnet.magnetic_induction(distance)

        # Определение пространственной производной B по оси
        delta = 1e-6  # Малое приращение для численной дифференциации
        dB_dx = (self.magnet.magnetic_induction(distance + delta) - B) / delta

        # Расчет ЭДС с учетом точного B и dB/dx
        dPhi_dt = -self.turns * self.area * dB_dx * magnet_velocity
        return dPhi_dt

    def get_eds(self, z_m, v_m, t):
        distance = z_m - self.coil_position(t)

        # Проверка, находится ли магнит внутри катушки
        if self.position <= z_m <= self.position + self.height:
            # Расчёт магнитной индукции из класса Magnet
            B = self.magnet.magnetic_induction(distance) 
            # Расчёт ЭДС по закону Фарадея
            delta = 1e-6  # Малое приращение для численной дифференциации
            dB_dx = (self.magnet.magnetic_induction(distance + delta) - B) / delta
            eds = -self.turns * self.area * dB_dx * v_m
            return eds
        else:
            return 0



def shaker_force(magnet, t):
    """
    Расчет силы, действующей на магнит при вибрации.
    """
    
    return -magnet.mass * W**2 * X0 * math.cos(W * t)