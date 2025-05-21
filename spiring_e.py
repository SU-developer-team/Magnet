# Импорт библиотек (так как среда была сброшена)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Задаем стандартные параметры пружины
D = 0.05  # м (диаметр витка)
d = 0.005  # м (диаметр проволоки)
G = 44e9  # Па (модуль сдвига для титана)
L = 0.1  # м (длина пружины)

# Параметры системы
m = 0.5  # кг (масса груза)
k = (d**4 * G * np.pi) / (8 * D**2 * L)  # Жесткость пружины
b = 0.05  # Коэффициент демпфирования (потери энергии)
y0 = 0.1  # Начальное удлинение (м)
v0 = 0  # Начальная скорость (м/с)

# Дифференциальное уравнение системы
def damped_oscillator(t, z):
    y, v = z
    dydt = v
    dvdt = - (b/m) * v - (k/m) * y
    return [dydt, dvdt]

# Временной интервал
t_span = (0, 50)
t_eval = np.linspace(0, 50, 300)

# Решение системы с методом LSODA (лучше сохраняет энергию)
sol_corrected = solve_ivp(
    damped_oscillator, t_span, [y0, v0], t_eval=t_eval, method='LSODA', rtol=1e-9, atol=1e-9
)

# Вычисляем энергии заново
y_vals_corrected = sol_corrected.y[0]
v_vals_corrected = sol_corrected.y[1]
E_p_corrected = 0.5 * k * y_vals_corrected**2  # Потенциальная энергия
E_k_corrected = 0.5 * m * v_vals_corrected**2  # Кинетическая энергия
E_total_corrected = E_p_corrected + E_k_corrected  # Полная энергия

# Построение исправленного графика
plt.figure(figsize=(10, 6))

plt.plot(sol_corrected.t, E_p_corrected, label="Потенциальная энергия", color="blue")
plt.plot(sol_corrected.t, E_k_corrected, label="Кинетическая энергия", color="green")
plt.plot(sol_corrected.t, E_total_corrected, label="Полная энергия (исправлено)", color="red", linestyle="dashed")

plt.xlabel("Время (с)")
plt.ylabel("Энергия (Дж)")
plt.title("Проверка сохранения энергии в системе")
plt.legend()
plt.grid(True)

plt.show()
