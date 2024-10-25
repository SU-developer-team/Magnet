import math
import numpy as np

G = 9.8015 # ускорение свободного падения (м/с^2) г. Алматы
X0 = 0.001 # амплитуда колебаний
W = 2 * math.pi * 10  # частота колебаний
F0 =  20 # сила магнитного const


class Magnet:
    def __init__(self, diameter, height, mass):
        self.radius = diameter / 2
        self.height = height
        self.mass = mass
        self.a = 19690.9170 # constant from the approximation of the magnetic force
        self.b = 0.5324 # constant from the approximation of the magnetic force

    def get_force(self, z):
        """
        Расчет силы магнита в точке z по оси магнита.
        """
        return F0 / (self.radius**2 + z**2)**1.5
    
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
    

def shaker_force(magnet, t):
    """
    Расчет силы, действующей на магнит при вибрации.
    """
    
    return magnet.mass * G * W**2 * X0 * math.sin(W * t)