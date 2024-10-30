import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import csv
from models.models import Magnet, Coil, calculate_emf

# Создаем объекты магнита и катушки с новыми размерами
magnet = Magnet(mass=0.043, diameter=0.008, height=0.01)  # Масса 43 г, диаметр 8 мм, высота 10 мм
coil = Coil(turns=50, radius=0.0004, position=0)  # Катушка в позиции z = 0

# Параметры движения магнита
def external_force(t, z, v):
    """
    Функция, возвращающая внешнюю силу в зависимости от времени, положения и скорости.
    Моделируем пружинную силу для создания гармонических колебаний.
    """
    k = 5.0  # Коэффициент жесткости пружины (Н/м)
    z_eq = 0  # Положение равновесия (м)
    print(f"ex f {k * (z - z_eq)}")
    return -k * (z - z_eq)

z0 = 0.05       # Начальное положение (м)
v0 = 0          # Начальная скорость (м/с)
t_total = 2     # Общее время моделирования (с)
time_step = 0.001  # Шаг по времени (с)

# Расчет движения магнита
times, positions, velocities, accelerations = magnet.motion(external_force, z0, v0, t_total, time_step)

# Расчет ЭДС
emfs = calculate_emf(magnet, coil, times, positions)

# Сохранение результатов в CSV-файл
with open('emf_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Time (s)', 'Position (m)', 'Velocity (m/s)', 'Acceleration (m/s^2)', 'EMF (V)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for t, z, v, a, emf in zip(times, positions, velocities, accelerations, emfs):
        writer.writerow({'Time (s)': t, 'Position (m)': z, 'Velocity (m/s)': v, 'Acceleration (m/s^2)': a, 'EMF (V)': emf})

# Настройка графики
# Настройка графики
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# График положения магнита и катушки
ax1.set_xlim(-0.012, 0.012)  # Устанавливаем границы для горизонтальной оси (новая ось X)
ax1.set_ylim(-0.06, 0.06)  # Устанавливаем границы для вертикальной оси (новая ось Y)
ax1.set_xlabel('Сечение (м)')
ax1.set_ylabel('Положение по оси z (м)')
ax1.set_title('Движение магнита через катушку')

# Устанавливаем равные масштабы осей
ax1.set_aspect('equal', adjustable='box')

# Отображение катушки как пружины
coil_length = 0.02  # Длина катушки (м)
num_coil_turns = 12  # Количество витков катушки для отображения
coil_diameter = 0.01 
# Создаем массив точек для катушки
z_coil = np.linspace(coil.position - coil_length / 2, coil.position + coil_length / 2, 1000)
radius = coil_diameter * 0.8  # Уменьшаем радиус для лучшего отображения

x_coil = radius * np.sin(2 * np.pi * num_coil_turns * (z_coil - (coil.position - coil_length / 2)) / coil_length)

# Рисуем катушку вертикально
ax1.plot(x_coil, z_coil, color='blue')

# Отображение магнита
magnet_height = magnet.height
magnet_patch = plt.Rectangle((-magnet.radius, positions[0] - magnet_height / 2), 2*magnet.radius, magnet_height, color='red', label='Магнит')
ax1.add_patch(magnet_patch)
ax1.legend()

# График ЭДС
ax2.set_xlim(times[0], times[-1])
ax2.set_ylim(min(emfs) * 1.1, max(emfs) * 1.1)
ax2.set_xlabel('Время (с)')
ax2.set_ylabel('ЭДС (В)')
ax2.set_title('Индуцированная ЭДС в катушке')
line_emf, = ax2.plot([], [], color='green')

# График скорости магнита
ax3.plot(times, velocities)
ax3.set_xlabel('Время (с)')
ax3.set_ylabel('Скорость (м/с)')
ax3.set_title('Скорость магнита во времени')

# Инициализация данных для анимации
emf_data = []
time_data = []

def animate(i):
    if i < len(times):
        # Обновление положения магнита (вертикально)
        z = positions[i]
        magnet_patch.set_y(z - magnet.height / 2)

        # Обновление графика ЭДС
        time_data.append(times[i])
        emf_data.append(emfs[i])
        line_emf.set_data(time_data, emf_data)

        return magnet_patch, line_emf
    else:
        return magnet_patch, line_emf

# Создание анимации
ani = FuncAnimation(fig, animate, frames=len(times), interval=20, blit=True)

plt.tight_layout()
plt.show()
# plt.savefig('/home/yerlan/projects/Magnet/emf_graph.png')
# ani.save('magnet_animation.gif', writer=PillowWriter(fps=30))