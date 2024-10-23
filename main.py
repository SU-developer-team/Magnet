import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from models import Magnet, Coil, calculate_emf

# Создаем объекты магнита и катушки с новыми размерами
magnet = Magnet(mass=0.043, diameter=0.008, height=0.01)  # Диаметр 8 мм, высота 10 мм
coil = Coil(turns=100, radius=0.01)  # Радиус 10 мм

# Задаем параметры движения магнита
velocity = 0.5 # м/с
z_start = -0.05  # Начальное положение магнита (м)
z_end = 0.05     # Конечное положение магнита (м)
time_step = 0.001  # Шаг по времени (с)
num_cycles = 1  # Количество циклов движения магнита

# Определяем параметры для фаз движения
velocities = [velocity, -velocity] * num_cycles
z_starts = [z_start, z_end] * num_cycles
z_ends = [z_end, z_start] * num_cycles

# Расчет ЭДС
times, emfs, positions = calculate_emf(magnet, coil, velocities, z_starts, z_ends, time_step)

# Сохранение результатов в CSV-файл
with open('emf_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Time (s)', 'EMF (V)', 'Position (m)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for t, emf, z in zip(times, emfs, positions):
        writer.writerow({'Time (s)': t, 'EMF (V)': emf, 'Position (m)': z})

# Настройка графики
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# График положения магнита и катушки
ax1.set_xlim(z_start*1.1, z_end*1.1)
ax1.set_ylim(-0.012, 0.012)
ax1.set_xlabel('Положение по оси z (м)')
ax1.set_ylabel('Сечение (м)')
ax1.set_title('Движение магнита через катушку')

# Устанавливаем равные масштабы осей
ax1.set_aspect('equal', adjustable='box')

# Отображение катушки как пружины
coil_length = 0.03  # Длина катушки (м)
num_coil_turns = 50  # Количество витков катушки для отображения

# Создаем массив точек для катушки
z_coil = np.linspace(coil.position - coil_length / 2, coil.position + coil_length / 2, 1000)
radius = coil.radius * 0.8  # Уменьшаем радиус для лучшего отображения

x_coil = radius * np.sin(2 * np.pi * num_coil_turns * (z_coil - (coil.position - coil_length / 2)) / coil_length)

# Рисуем катушку
ax1.plot(z_coil, x_coil, color='blue')

# Отображение магнита
magnet_height = magnet.height
magnet_patch = plt.Rectangle((z_start - magnet_height/2, -magnet.radius), magnet_height, 2*magnet.radius, color='red', label='Магнит')
ax1.add_patch(magnet_patch)
ax1.legend()

# График ЭДС
ax2.set_xlim(times[0], times[-1])
ax2.set_ylim(min(emfs) * 1.1, max(emfs) * 1.1)
ax2.set_xlabel('Время (с)')
ax2.set_ylabel('ЭДС (В)')
ax2.set_title('Индуцированная ЭДС в катушке')

# Инициализация линии графика ЭДС
line_emf, = ax2.plot([], [], color='green')

# Инициализация списков для анимации
emf_data = []
time_data = []

# Список цветов для циклов
# colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown']

def animate(i):
    # Определяем текущий цикл
    total_frames = len(times)
    cycles = len(velocities)
    frames_per_cycle = total_frames // cycles
    cycle = i // frames_per_cycle

    # Обновляем положение магнита
    z = positions[i]
    magnet_patch.set_x(z - magnet.height / 2)

    # Обновление графика ЭДС
    time_data.append(times[i])
    emf_data.append(emfs[i])
    line_emf.set_data(time_data, emf_data)

    return magnet_patch, line_emf

# Общее количество кадров в анимации
total_frames = len(times)

# Создание анимации
ani = FuncAnimation(fig, animate, frames=total_frames, interval=20, blit=True)

plt.tight_layout()
plt.show()
