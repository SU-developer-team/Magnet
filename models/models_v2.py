import math
import numpy as np
 

class Magnet:
    def __init__(self, diameter, height, mass):
        self.radius = diameter / 2
        self.height = height
        self.mass = mass
        self.diameter = diameter
        
        # self.a = 7.8614069  
        self.a = 0.007861  
        self.b = -0.121755

    def get_force(self, x, F_shaker):
        """
        Расчет силы магнита в точке z по оси магнита.
        """ 
        mu_0 = 4 * np.pi * 1e-7  # Магнитная проницаемость вакуума 
        Br = 0.648371 # Тл
        b = 0.704375

        term1 = (2 * (self.height + x)) / np.sqrt((self.height + x)**2 + self.radius**2)
        term2 = (2 * self.height + x) / np.sqrt((2 * self.height + x)**2 + self.radius**2)
        term3 = x / np.sqrt(x**2 + self.radius**2)
        
        force = (np.pi * Br**2 * self.radius**2) / (2 * mu_0) * (term1 - term2 - term3) + b
        return force + F_shaker

        return self.a / z + self.b + F_shaker
  
    
    def magnetic_induction(self, z):
        """
        Расчет магнитной индукции в точке z по оси магнита.
        z - расстояние от центра магнита вдоль оси.
        """
        Br = 0.26  # Остаточная магнитная индукция магнита (Тл)
        R = self.radius
        L = self.height / 2

        def Bz(z, R, L):
            return (Br / 2) * (
                (z + L) / math.sqrt(R**2 + (z + L)**2) -
                (z - L) / math.sqrt(R**2 + (z - L)**2)
            )

        return Bz(z, R, L)


class Coil:
    def __init__(self, turns_count, thickness, radius, position, magnet, wire_diameter=0.0005, layer_count=4):
        self.turns_count = turns_count  # Общее число витков
        self.radius = radius
        self.position = position  # Базовое положение катушки по оси z
        self.magnet = magnet
        self.thickness = thickness  # Толщина катушки (м)
        self.wire_diameter = wire_diameter  # Диаметр провода (м)
        self.layer_count = layer_count  # Количество слоев проводов (обмоток друг над другом)

        # Расчет числа витков на слой
        self.turns_per_layer = int(self.turns_count / self.layer_count)

        # Проверка и корректировка общего числа витков
        self.turns_count = self.turns_per_layer * self.layer_count

        # Вычисление толщины катушки (с учетом слоев)
        calculated_thickness = self.layer_count * self.wire_diameter
        if calculated_thickness > self.thickness:
            raise ValueError("Заданная толщина меньше, чем требуется для размещения заданного числа слоев.")

        # Вычисление высоты катушки (длина вдоль оси z)
        self.height = self.turns_per_layer * self.wire_diameter

        # Создаем список для хранения информации о каждой витке
        self.turns = []
        self.initialize_turns()

    def initialize_turns(self):
        """
        Инициализирует позиции и параметры каждой витки в катушке.
        """
        for layer in range(self.layer_count):
            # Радиус каждого слоя увеличивается на диаметр провода
            layer_radius = self.radius + layer * self.wire_diameter
            for turn in range(self.turns_per_layer):
                # Позиция каждой витки вдоль оси z (вертикальное расположение)
                turn_z_position = self.position + turn * self.wire_diameter
                self.turns.append({
                    'layer': layer,
                    'radius': layer_radius,
                    'z_position': turn_z_position
                })

    def coil_position(self, shaker, t):
        """
        Возвращает базовое положение катушки в момент времени t с учетом колебаний шейкера.
        """
        return self.position + shaker.X0 * np.sin(shaker.W * t)

    def get_eds(self, shaker, magnet_position, magnet_velocity, t):
        """
        Рассчитывает ЭДС для каждой витки и возвращает список ЭДС витков и их суммарное значение.
        """
        total_eds = 0
        eds_per_turn = []

        # Базовое положение катушки с учетом движения шейкера
        coil_base_position = self.coil_position(shaker, t)

        for turn in self.turns:
            # Положение витки с учетом базового положения катушки
            turn_position = coil_base_position + (turn['z_position'] - self.position)

            # Расстояние от магнита до витки
            distance = magnet_position - turn_position

            # Магнитная индукция в точке витки
            B = self.magnet.magnetic_induction(distance)

            # Численная производная магнитной индукции по оси z
            delta = 1e-6  # Малое приращение для численной дифференциации
            B_plus = self.magnet.magnetic_induction(distance + delta)
            dB_dx = (B_plus - B) / delta

            # Площадь витки (можно учитывать изменение площади для каждого слоя)
            turn_area = math.pi * turn['radius']**2

            # Расчет ЭДС для витки
            eds = -dB_dx * magnet_velocity * turn_area

            # Добавляем ЭДС витки в список
            eds_per_turn.append(eds)
            
            # Новая версия формулы
            # B0 = 0.0000000000000001
            # L = 1.2
            # a = magnet_velocity / t
            # R = 1.5
            # eds = (magnet_velocity - L*a/R)  * turn_area * B0 / distance**2

            # Суммируем ЭДС
            total_eds += eds

        return eds_per_turn, total_eds


class Shaker:
    def __init__(self, G, miew, X0):
        self.G = G,
        self.miew = miew
        self.X0 = X0
        self.W = 2 * math.pi * miew 

    def get_force(self, magnet, t):
        """
        Расчет силы, действующей на магнит при вибрации.
        """
        return -magnet.mass * self.W**2 * self.X0 * math.cos(self.W * t)