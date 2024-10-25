import math
import numpy as np

G = 9.8015 # ускорение свободного падения (м/с^2) г. Алматы
X0 = 0.002 # амплитуда колебаний
W = 2 * math.pi * 10  # частота колебаний
F0 =  0.000009 # сила магнитного const
a = 0.00009
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
        return a / z**2
        # return a / (self.radius**2 + z**2)**1.5
    
    # def 
    
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


class Coil:
    def __init__(self, turns, radius, position=0):
        self.turns = turns
        self.radius = radius
        self.area = math.pi * radius**2
        self.position = position  # Положение катушки по оси z 

    def coil_position(self, t):
        """
        Возвращает положение катушки в момент времени t с учетом колебаний шейкера.
        """
        return self.position + X0 * np.sin(W * t)

    def get_eds(self, magnet_position, magnet_velocity, t):
        """
        Рассчитывает ЭДС, индуцированную в катушке, в зависимости от положения и скорости магнита.
        """
        # Расстояние между магнитом и катушкой
        distance = abs(magnet_position - self.coil_position(t))

        # Простейшая модель магнитного поля, убывающего с расстоянием, например, B ~ 1/distance^2
        B = 1 / distance**2 if distance > 0 else 0  # Избегаем деления на ноль

        # Производная магнитного потока по времени для ЭДС
        dPhi_dt = -self.turns * self.area * B * magnet_velocity

        return dPhi_dt

def shaker_force(magnet, t):
    """
    Расчет силы, действующей на магнит при вибрации.
    """
    
    return -magnet.mass * W**2 * X0 * math.cos(W * t)