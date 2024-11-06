import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Гиперболическая функция для аппроксимации
def hyperbola_func(x, a, b):
    return a / x + b

def plot_data_with_hyperbola_approximation():
    csv_file = 'data.csv'
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

    # Подгонка данных под гиперболическую функцию
    popt, pcov = curve_fit(hyperbola_func, h_values, avg_values, p0=(1, 0))
    a, b = popt
    print(f"Найденные параметры аппроксимации: a = {a:.6f}, b = {b:.6f}")

    # Генерация точек для аппроксимированной кривой
    h_approx = np.linspace(min(h_values), max(h_values), 500)
    avg_approx = hyperbola_func(h_approx, *popt)

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

    # Построение аппроксимированной кривой (гипербола)
    plt.plot(h_approx, avg_approx, label=f'Гиперболическая аппроксимация (a={a:.4f}, b={b:.4f})', color='red')

    # Добавление заголовков и легенды
    plt.title('Гиперболическая аппроксимация для среднего значения b от h')
    plt.xlabel('Расстояние между магнитами (м)')
    plt.ylabel('Средняя масса (кг)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    plot_data_with_hyperbola_approximation()

# Запуск основной функции
if __name__ == "__main__":
    main()
