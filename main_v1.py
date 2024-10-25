import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import csv
from models import Magnet, Coil, calculate_emf

# Создаем объекты магнита и катушки с новыми размерами
magnet = Magnet(mass=0.043, diameter=0.008, height=0.01)  # Увеличенные F0 и f0
coil = Coil(turns=100, radius=0.01, position=0)  # Катушка в позиции z = 0

def magnetic_force(z):
    epsilon = 1e-6  # Маленькое значение для предотвращения деления на ноль
    h = 0.1  # Расстояние между магнитом и катушкой
    f0=1e-10
    f1=1e-10

    z_clipped = np.clip(z, epsilon, h - epsilon)
    f = f0 / (h - z_clipped)**2 - f1 / z_clipped**2
    f = np.clip(f, -1e6, 1e6)  # Ограничиваем силу, чтобы избежать больших значений
    return f

def external_force(t, z, v):
    # damping_coefficient = 0.01  # Коэффициент затухания
    
    shaker_amplitude=0.01
    shaker_frequency=2
    shaker_phase=0
    
    external_oscillation = shaker_amplitude * np.sin(shaker_frequency * t + shaker_phase)
    f = magnetic_force(z) * v + external_oscillation
    return f

# Параметры движения магнита
z0 = 0.4 # Начальное положение (м)
v0 = 0.0  # Начальная скорость (м/с)
t_total = 2  # Общее время моделирования (с)
time_step = 0.001  # Шаг по времени (с)
z_min=-0.5
z_max=0.5

# Расчет движения магнита
times, positions, velocities, accelerations = magnet.motion(lambda t, z, v: external_force(t, z, v), z0, v0, t_total, time_step, z_min, z_max)

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
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# График положения магнита и катушки
ax1.set_xlim(-0.3, 0.3)  # Устанавливаем границы для горизонтальной оси (новая ось X)
ax1.set_ylim(z_min*1.1, z_max*1.1)  # Устанавливаем границы для вертикальной оси (новая ось Y)
ax1.set_xlabel('Сечение (м)')
ax1.set_ylabel('Положение по оси z (м)')
ax1.set_title('Движение магнита через катушку')

# Устанавливаем равные масштабы осей
ax1.set_aspect('equal', adjustable='box')

# Отображение катушки как пружины
coil_length = 0.15  # Длина катушки (м)
num_coil_turns = 10  # Количество витков катушки для отображения

# Создаем массив точек для катушки
z_coil = np.linspace(coil.position - coil_length / 2, coil.position + coil_length / 2, 1000)
radius = coil.radius * 12  # Уменьшаем радиус для лучшего отображения

x_coil = radius * np.sin(2 * np.pi * num_coil_turns * (z_coil - (coil.position - coil_length / 2)) / coil_length)

# Рисуем катушку вертикально
ax1.plot(x_coil, z_coil, color='blue')

# Отображение магнита
magnet_height = magnet.height
magnet_patch = plt.Rectangle((-magnet.radius, positions[0] - magnet_height / 10), 10*magnet.radius,  10*magnet_height, color='red', label='Магнит')
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
