import os
import glob
import pandas as pd
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

# Построение графика
plt.figure(figsize=(12, 8))

for df in data_list:
    z = df['z']
    s = df['s']
    μ = df['μ'].iloc[0]
    plt.plot(z, s, marker='', linestyle='-', label=f'μ = {μ}')

plt.title('Зависимость s от z при разных μ')
plt.xlabel('z (Высота верхнего магнита)')
plt.ylabel('s (Среднеквадратическое значение ЭДС)')
plt.legend(title='μ (Частота)', loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('z_vs_s_for_different_mu.png')  # Сохранение графика в файл
plt.show()
