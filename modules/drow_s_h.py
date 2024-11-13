import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Чтение данных из output.csv
data = pd.read_csv('../z_h/csv/s_z__20.csv')

# Построение графика
plt.figure(figsize=(20, 6))
plt.plot(data['z'], data['s'], marker='o', linestyle='-', color='b', label='s(z)')

# Настройка графика
plt.title('Зависимость s от z')
plt.xlabel('z (Высота верхнего магнита)')
plt.ylabel('s (Среднеквадратическое значение ЭДС)')
plt.xticks(np.arange(0.04, 0.91, 0.05))
plt.legend()
plt.grid(True)
plt.savefig('../z_h/png/s_z__20.png')
# Сохранение и отображение графика
# plt.savefig('plot_output.png')  # Сохранить график как PNG файл
plt.show()                       # Показать график на экране
