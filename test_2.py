from scipy import stats
import numpy as np

def get_student_t(df, confidence=0.95):
    t_table = {
        1: 12.71, 2: 4.30, 3: 3.18, 4: 2.77, 5: 2.57,
        6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13,
        30: 2.04, 40: 2.02, 60: 2.00, 120: 1.980
    }
    return t_table.get(df, t_table[120])  # fallback


# Исходные данные (граммы)
raw_data = np.array([
    [55, 12, 13, 11, 13, 13],
    [50, 18, 18, 17, 17, 16],
    [45, 26, 25, 24, 24, 25],
    [40, 32, 36, 35, 38, 37],
    [35, 49, 48, 49, 47, 48],
    [30, 60, 68, 79, 77, 81],
    [25, 110, 117, 119, 118, 115],
    [20, 203, 192, 198, 195, 204],
    [15, 355, 353, 341, 362, 348],
    [10, 582, 621, 609, 623, 626],
])

# Преобразование
g = 9.81
distances = raw_data[:, 0] / 1000
masses = raw_data[:, 1:]
n = masses.shape[1]
alpha = 0.05
t_coef = get_student_t(n - 1, confidence=0.95) 

# Вычисления
results = []
for i in range(len(distances)):
    mean_mass = np.mean(masses[i]) / 1000
    std_mass = np.std(masses[i], ddof=1) / 1000
    mean_force = mean_mass * g
    std_force = std_mass * g
    margin = t_coef * std_force / np.sqrt(n)
    ci_lower = mean_force - margin
    ci_upper = mean_force + margin
    results.append((distances[i], mean_force, ci_lower, ci_upper))

import pandas as pd 
df = pd.DataFrame(results, columns=["Дистанция (м)", "Средняя сила (Н)", "CI нижний", "CI верхний"]) 
print(df.to_string(index=False))  
df.to_csv("confidence_intervals.csv", index=False)
print("Таблица сохранена в confidence_intervals.csv")
