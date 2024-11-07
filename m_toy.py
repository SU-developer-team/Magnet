import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Параметры магнита и катушки
mu0 = 4 * np.pi * 1e-7  # магнитная постоянная (H/m)
Br = 1.2  # остаточная магнитная индукция магнита (T)
radius_magnet = 0.01  # радиус магнита (м)
height_magnet = 0.02  # высота магнита (м)
mass_magnet = 0.05  # масса магнита (кг)
resistance_coil = 1.0  # сопротивление катушки (Ом)
num_turns_coil = 100  # число витков катушки
area_turn = np.pi * radius_magnet**2  # площадь витка катушки (м²)

# Функция для расчета магнитного поля вдоль оси магнита
def magnetic_field(z, Br, radius, height):
    z1 = z + height / 2
    z2 = z - height / 2
    Bz = (Br / 2) * ((z1 / np.sqrt(radius**2 + z1**2)) - (z2 / np.sqrt(radius**2 + z2**2)))
    return Bz

# Численное вычисление производной магнитного поля dB/dz
def magnetic_field_gradient(z, Br, radius, height, delta=1e-6):
    Bz = magnetic_field(z, Br, radius, height)
    Bz_plus = magnetic_field(z + delta, Br, radius, height)
    dB_dz = (Bz_plus - Bz) / delta
    return dB_dz

# Функция для расчета ускорения магнита
def magnet_acceleration(z, v, mass, Br, radius, height, num_turns, area, resistance):
    dB_dz = magnetic_field_gradient(z, Br, radius, height)
    emf = -num_turns * area * dB_dz * v  # закон Фарадея
    current = emf / resistance  # ток в катушке
    force = num_turns * area * current * dB_dz  # сила, действующая на магнит
    acceleration = force / mass
    return acceleration

# Уравнения движения магнита
def equations(y, t, mass, Br, radius, height, num_turns, area, resistance):
    z, v = y
    dzdt = v
    dvdt = magnet_acceleration(z, v, mass, Br, radius, height, num_turns, area, resistance)
    return [dzdt, dvdt]

# Начальные условия и параметры моделирования
initial_position = 0.0  # начальное положение магнита (м)
initial_velocity = 1.0  # начальная скорость магнита (м/с)
y0 = [initial_position, initial_velocity]
time = np.linspace(0, 1, 1000)  # время моделирования (с)

# Решение уравнений движения
solution = odeint(equations, y0, time, args=(mass_magnet, Br, radius_magnet, height_magnet, num_turns_coil, area_turn, resistance_coil))

# Расчет ЭДС
positions = solution[:, 0]
velocities = solution[:, 1]
emfs = -num_turns_coil * area_turn * np.array([magnetic_field_gradient(z, Br, radius_magnet, height_magnet) for z in positions]) * velocities

# Анимация движения магнита
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# График положения магнита и его катушки
ax1.set_xlim(0, 0.1)
ax1.set_ylim(-0.02, 0.02)
ax1.set_title("Движение магнита в катушке")
position_line, = ax1.plot([], [], label="Положение магнита (z)", color="blue")
ax1.set_xlabel("Время (с)")
ax1.set_ylabel("Положение (м)")
ax1.legend()

# График ЭДС
ax2.set_xlim(0, 0.1)
ax2.set_ylim(np.min(emfs) * 1.1, np.max(emfs) * 1.1)
ax2.set_title("ЭДС, индуцированная катушкой")
emf_line, = ax2.plot([], [], label="ЭДС (V)", color="red")
ax2.set_xlabel("Время (с)")
ax2.set_ylabel("ЭДС (В)")
ax2.legend()

# Функция инициализации для анимации
def init():
    position_line.set_data([], [])
    emf_line.set_data([], [])
    return position_line, emf_line

# Функция обновления для анимации
def update(frame):
    t = time[:frame]
    position_line.set_data(t, positions[:frame])
    emf_line.set_data(t, emfs[:frame])
    return position_line, emf_line

# Создание анимации
ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True)

# Показываем анимацию
plt.tight_layout()
plt.show()
