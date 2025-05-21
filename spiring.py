
import numpy as np
import matplotlib.pyplot as plt

# Задаем стандартные параметры пружины
D = 0.05  # м (диаметр витка)
d = 0.005  # м (диаметр проволоки)
# G = 80e9  # Па (модуль сдвига для стали)
G = 44e9  # Па (модуль сдвига для титана)
L = 0.1  # м (длина пружины)

# Генерируем диапазон значений удлинения y
y_values = np.linspace(0.001, 0.1, 100)

# Вычисляем F по упрощенной формуле
F_values = (y_values * d**4 * G * np.pi) / (8 * D**2 * L)

# Построение графика
plt.figure(figsize=(8, 5))
plt.plot(y_values, F_values, label=r'$F = \frac{y \cdot d^4 G \pi}{8 D^2 L}$', color='g')

# Оформление графика
plt.xlabel('Удлинение пружины y (м)')
plt.ylabel('Сила упругости F (Н)')
plt.title('Зависимость силы упругости F от удлинения y')
plt.legend()
plt.grid(True)

# Показать график
plt.show()
