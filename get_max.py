import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Укажите директорию, где находятся ваши CSV файлы
csv_dir = 'z_s_center_coil_position/csv/'

# Используем glob для получения списка всех CSV файлов, соответствующих шаблону
csv_files = glob.glob(os.path.join(csv_dir, 's_z__*.csv'))

# Список для хранения данных из всех файлов
data_list = []

def extract_mu(filename):
    """
    Функция для извлечения значения μ из имени файла.
    Ожидается, что имя файла имеет формат 's_z__{μ}.csv'
    """
    basename = os.path.basename(filename)
    # Удаляем префикс и суффикс
    mu_str = basename.replace('s_z__', '').replace('.csv', '').replace('с', '')
    try:
        μ = float(mu_str)
        return μ
    except ValueError:
        return None

# Читаем данные из каждого файла и добавляем в список
for file in csv_files:
    μ = extract_mu(file)
    if μ is not None:
        df = pd.read_csv(file)
        df['μ'] = μ  # Добавляем столбец с μ
        data_list.append(df)

# Сортируем список данных по значению μ для упорядоченного отображения
data_list.sort(key=lambda df: df['μ'].iloc[0])

# Целевое значение z
z_target = 0.135

# Списки для хранения μ и соответствующих s при z_target
mu_values = []
s_values_at_z_target = []

# Проходим по каждому DataFrame и извлекаем s при z = z_target
for df in data_list:
    z = df['z']
    s = df['s']
    μ = df['μ'].iloc[0]

    # Проверяем, есть ли z_target в z
    if z_target in z.values:
        s_value = s[z == z_target].iloc[0]
    else:
        # Если z_target нет в z, интерполируем значение s
        s_value = np.interp(z_target, z, s)

    mu_values.append(μ)
    s_values_at_z_target.append(s_value)

    print(f"μ = {μ}, s({z_target}) = {s_value}")

# Построение графика s от μ при z = z_target
plt.figure(figsize=(10, 6))
plt.plot(mu_values, s_values_at_z_target, marker='o', linestyle='-')
plt.title(f'Зависимость s от μ при z = {z_target}')
plt.xlabel('μ (Частота)')
plt.ylabel(f's (Среднеквадратическое значение ЭДС при z = {z_target})')
plt.grid(True)
plt.xticks(np.arange(0, 105, 5))
plt.tight_layout()
plt.savefig(f's_vs_mu_at_z_{z_target}.png')  # Сохранение графика в файл
plt.show()
