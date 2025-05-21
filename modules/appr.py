import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Определение функции магнитного поля H(x)
def magnetic_field(x, Br, b):
    mu_0 = 4 * np.pi * 1e-7  # Магнитная проницаемость вакуума
    R = 0.0195 / 2  # Радиус магнита в метрах (например, 9.75 мм)
    L = 0.01  # Длина магнита в метрах (например, 10 мм)
    
    term1 = (x + L) / np.sqrt(R**2 + (x + L)**2)
    term2 = x / np.sqrt(R**2 + x**2)
    return (Br / (2 * mu_0)) * (term1 - term2) + b

def plot_data_with_magnetic_field_approximation():
    csv_file = r'D:\PROJECTs\leaves_detection\magnet\Magnet_clean\data.csv'
    h_values = []
    avg_values = []

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
            avg = (m1 + m2 + m3) / 3
            h_values.append(h)
            avg_values.append(avg)

    # Преобразуем данные в numpy массивы
    h_values = np.array(h_values)
    avg_values = np.array(avg_values)

    # Подгонка данных под функцию магнитного поля H(x) с границами для Br
    initial_guess = [0.000011, 0.0]  # Начальные значения для Br и b
    bounds = ([0.00001087, -np.inf], [0.000013, np.inf])  # Ограничения для Br и b

    popt, pcov = curve_fit(magnetic_field, h_values, avg_values, p0=initial_guess, bounds=bounds)
    Br, b = popt
    print(f"Найденные значения: Br = {Br:.6f} Тл, b = {b:.6f}")

    # Генерация точек для аппроксимированной кривой
    h_approx = np.linspace(min(h_values), max(h_values), 500)
    avg_approx = magnetic_field(h_approx, Br, b)

    # Вычисление средней относительной ошибки по ближайшим точкам
    relative_errors = 0.0
    for i in range(len(avg_values)):
        # Находим индекс ближайшего значения в avg_approx к avg_values[i]
        idx = np.argmin(np.abs(avg_approx - avg_values[i]))
        closest_approx_value = avg_approx[idx]
        
        # Вычисляем относительную ошибку между avg_values[i] и closest_approx_value
        rel_error = np.abs((avg_values[i] - closest_approx_value) / avg_values[i])
        relative_errors += rel_error

    # Вычисление средней относительной ошибки в процентах
    relative_errors_percent = (relative_errors / len(avg_values)) * 100
    print(f"Средняя относительная ошибка: {relative_errors_percent:.2f}%")

    # Построение исходного графика
    plt.plot(h_values, avg_values, 'o', label=f'Исходные данные. Средняя относительная ошибка {relative_errors_percent:.2f}%')

    # Построение аппроксимированной кривой (магнитное поле)
    plt.plot(h_approx, avg_approx, label=f'Аппроксимация магнитным полем (Br={Br:.4f} Тл, b={b:.4f})', color='red')

    # Добавление заголовков и легенды
    plt.title('Аппроксимация магнитным полем для среднего значения b от h')
    plt.xlabel('Расстояние между магнитами (м)')
    plt.ylabel('Средняя масса (кг)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    plot_data_with_magnetic_field_approximation()

# Запуск основной функции
if __name__ == "__main__":
    main()
