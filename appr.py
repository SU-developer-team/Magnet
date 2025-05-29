import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

# Константы и параметры магнита
mu_0 = 4 * np.pi * 1e-7
R = 0.0195 / 2
L = 0.01

# --- Формулы --- #
def hyperbolla(x, Br, b): 
    return Br / x + b

def exponential(x, Br, b):
    return Br **5 * np.exp(-x / L)*1 + b

def castaner(x, Br, b):
    return (3 * np.pi * Br**2 * L**2 * R**4) / (2 * mu_0 * x**4 + 1e-12) + b

def furlani(x, Br, b):
    return (np.pi * Br**2 * R**4) / (4 * mu_0) * (
        1 / (x**2 + 1e-12) +
        1 / ((2 * L + x)**2 + 1e-12) -
        2 / ((L + x)**2 + 1e-12)
    ) + b

def furlani_zurek(x, Br, b):
    c = (4 * R) / 5
    x_c = x + c
    return (np.pi * Br**2 * R**4) / (4 * mu_0) * (
        1 / (x_c**2 + 1e-12) +
        1 / ((2 * L + x_c)**2 + 1e-12) -
        2 / ((L + x_c)**2 + 1e-12)
    ) + b

def cheedket(x, Br, b):
    term1 = 2 * (L + x) / np.sqrt((L + x)**2 + R**2)
    term2 = (2 * L + x) / np.sqrt((2 * L + x)**2 + R**2)
    term3 = x / np.sqrt(x**2 + R**2)
    return (np.pi * Br**2 * R**2) / (2 * mu_0) * (term1 - term2 - term3) + b

# --- Загрузка данных --- #

def read_csv(filename):
    h_values, avg_values = [], []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 6: continue
            h, *masses = map(float, row)
            h /= 1000
            avg_mass = np.mean([m / 1000 for m in masses])
            F = avg_mass * 9.81
            h_values.append(h)
            avg_values.append(F)
    return np.array(h_values), np.array(avg_values)

# --- Обёртка подгонки и визуализации --- #

def fit_and_plot(h, F, models, h1=None, F1=None):
    h_plot = np.linspace(min(h), max(h), 500)
    plt.figure(figsize=(12, 7))
    plt.plot(h, F, 'ko', label="Эксперимент", color='red')
    if h1 is not None and F1 is not None:
        plt.plot(h1, F1, '--', label="Эксперимент 2", color='blue')

    for name, model in models.items():
        try:
            popt, _ = curve_fit(model, h, F, p0=[1.2, 0.0], bounds=([0.0, -np.inf], [2.0, np.inf]))
            F_fit = model(h_plot, *popt)
            err = np.mean(np.abs((F - model(h, *popt)) / F)) * 100
            print(f"{name:<25} | Br = {popt[0]:.4f}, b = {popt[1]:.4f}, ошибка = {err:.2f}%")
            plt.plot(h_plot, F_fit, label=f"{name} ({err:.2f}%)")
        except Exception as e:
            print(f"{name} — ошибка подгонки: {e}")

    plt.xlabel("Расстояние (м)")
    plt.ylabel("Сила (Н)")
    plt.title("Сравнение моделей силы взаимодействия между магнитами")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Главная функция --- #

def main():
    h, F = read_csv('data_re.csv')
    h1, F1 = read_csv('data_re1.csv')
    models = {
        # "Castañer": castaner,
        # "Furlani": furlani,
        # "Furlani + Zurek": furlani_zurek,
        # "Гиперболла": hyperbolla,
        # "Экспоненциальная": exponential,
        # "Cheedket ": cheedket
    }
    fit_and_plot(h, F, models, h1, F1)

if __name__ == "__main__":
    main()
