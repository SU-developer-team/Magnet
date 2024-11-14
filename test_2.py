import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Определение функции силы в виде F(x) = a/x + b
def force_func(x, a, b):
    return a / x + b

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
            h, m1_raw, m2_raw, m3_raw = map(float, row)
            h = h / 1000  # Преобразование h в метры

            # Преобразование измерений силы
            m1 = (m1_raw - 32) / 1000
            m2 = (m2_raw - 32) / 1000
            m3 = (m3_raw - 32) / 1000
            g = 9.80665  # Ускорение свободного падения (м/с^2)
            F1 = m1 * g
            F2 = m2 * g
            F3 = m3 * g
            avg_force = (F1 + F2 + F3) / 3
            h_values.append(h)
            force_values.append(avg_force)

    # Преобразуем данные в numpy массивы
    h_values = np.array(h_values)
    force_values = np.array(force_values)

    # Начальное приближение для параметров a и b
    initial_guess = [1, 0]  # Подбор параметров a и b

    # Подгонка данных с использованием curve_fit
    popt, pcov = curve_fit(force_func, h_values, force_values, p0=initial_guess)
    
    # Вывод найденных значений параметров
    a_opt, b_opt = popt
    print(f"Найденное значение a: {a_opt:.6f}")
    print(f"Найденное значение b: {b_opt:.6f}")
    
    # Вычисление ошибки подгонки для параметров a и b
    perr = np.sqrt(np.diag(pcov))
    a_err, b_err = perr
    print(f"Погрешность определения a: ±{a_err:.6f}")
    print(f"Погрешность определения b: ±{b_err:.6f}")
    
    # Генерация точек для аппроксимированной кривой
    h_approx = np.linspace(min(h_values), max(h_values), 500)
    force_approx = force_func(h_approx, *popt)
    
    # Вычисление относительной ошибки
    relative_errors = np.abs((force_values - force_func(h_values, *popt)) / force_values)
    relative_errors_percent = np.mean(relative_errors) * 100
    print(f"Средняя относительная ошибка: {relative_errors_percent:.2f}%")
    
    # Построение исходного графика
    plt.plot(h_values, force_values, 'o', label='Экспериментальные данные')
    
    # Построение аппроксимированной кривой
    plt.plot(h_approx, force_approx, label=f'Модель (a={a_opt:.4f}, b={b_opt:.4f})', color='red')
    
    # Добавление заголовков и легенды
    plt.title('Сила в зависимости от расстояния x')
    plt.xlabel('x (м)')
    plt.ylabel('Сила F (Н)')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Сохранение графика
    plt.savefig("force_vs_x.png")

def main():
    plot_data_with_force_approximation()

if __name__ == "__main__":
    main()
