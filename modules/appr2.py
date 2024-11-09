import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Определение функции силы F(x) по формуле Cheedket с дополнительным параметром b
def force_cheedket(x, Br, b):
    mu_0 = 4 * np.pi * 1e-7  # Магнитная проницаемость вакуума
    R = 0.0195 / 2  # Радиус магнита в метрах (например, 9.75 мм)
    L = 0.01  # Длина магнита в метрах (например, 10 мм)
    
    term1 = (2 * (L + x)) / np.sqrt((L + x)**2 + R**2)
    term2 = (2 * L + x) / np.sqrt((2 * L + x)**2 + R**2)
    term3 = x / np.sqrt(x**2 + R**2)
    
    force = (np.pi * Br**2 * R**2) / (2 * mu_0) * (term1 - term2 - term3) + b
    return force

def plot_data_with_force_approximation():
    csv_file = 'data.csv'
    h_values = []
    force_values = []

    # Чтение данных из CSV
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 4:
                continue  # Пропуск строк с некорректными данными
            h, m1, m2, m3 = map(float, row)
            h /= 1000  # Преобразуем h в метры
            m1 = (m1 - 32) / 1000  # Преобразуем массу в кг
            m2 = (m2 - 32) / 1000
            m3 = (m3 - 32) / 1000
            avg_mass = (m1 + m2 + m3) / 3
            force_values.append(avg_mass * 9.81)  # Перевод массы в силу (Ньютон)
            h_values.append(h)

    # Преобразуем данные в numpy массивы
    h_values = np.array(h_values)
    force_values = np.array(force_values)

    # Подгонка данных под формулу Cheedket для силы с границами для Br и b 

    popt, pcov = curve_fit(force_cheedket, h_values, force_values)
    Br, b = popt
    print(f"Найденные значения: Br = {Br:.6f} Тл, b = {b:.6f}")

    # Генерация точек для аппроксимированной кривой
    h_approx = np.linspace(min(h_values), max(h_values), 500)
    force_approx = force_cheedket(h_approx, Br, b)

    # Вычисление средней относительной ошибки
    relative_errors = np.abs((force_values - force_cheedket(h_values, Br, b)) / force_values)
    mean_relative_error_percent = np.mean(relative_errors) * 100
    print(f"Средняя относительная ошибка: {mean_relative_error_percent:.2f}%")

    # Построение исходного графика
    plt.plot(h_values, force_values, 'o', label=f'Исходные данные. Средняя относительная ошибка {mean_relative_error_percent:.2f}%')

    # Построение аппроксимированной кривой (сила отталкивания)
    plt.plot(h_approx, force_approx, label=f'Аппроксимация силы (Br={Br:.4f} Тл, b={b:.4f})', color='red')

    # Добавление заголовков и легенды
    plt.title('Аппроксимация силы отталкивания между магнитами по формуле Cheedket')
    plt.xlabel('Расстояние между магнитами (м)')
    plt.ylabel('Сила (Н)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    plot_data_with_force_approximation()

# Запуск основной функции
if __name__ == "__main__":
    main()
