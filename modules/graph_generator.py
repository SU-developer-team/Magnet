import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import os
from scipy.ndimage import uniform_filter1d
import re


def extract_number(file_name):
    numbers = re.findall(r'/d+', file_name)
    return int(numbers[0]) if numbers else float('inf')  # Default to infinity if no number is found

def smooth_data(data, window_size=5):
    return uniform_filter1d(data, size=window_size)

class GraphGenerator:
    def __init__(self, csv_dir, save_dir, smooth=True, divide=1):
        """
        Инициализация объекта для генерации графиков.
        
        :param csv_dir: Путь к директории с CSV файлами.
        :param save_dir: Путь к директории для сохранения графиков.
        :param smooth: Флаг, указывающий нужно ли сглаживать данные.
        """
        self.csv_dir = csv_dir
        self.save_dir = save_dir
        self.smooth = smooth
        self.divide = divide

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def get_s(self, csv_file_path, q, draw=True):
        """
        Генерация графика на основе CSV файла и вычисление значения S.
        
        :param csv_file_path: Путь к CSV файлу.
        :param q: Имя графика (используется для сохранения файла).
        :return: Вычисленное значение S.
        """
        x_values = []
        y_values = []

        # Чтение данных из CSV файла
        with open(csv_file_path, 'r') as csvfile:
            # Определяем, есть ли заголовок
            has_header = csv.Sniffer().has_header(csvfile.read(1024))
            csvfile.seek(0)  # Возвращаемся в начало файла
            csvreader = csv.reader(csvfile)

            if has_header:
                next(csvreader)  # Пропускаем заголовок

            for row in csvreader:
                if len(row) == 2:
                    try:
                        x_values.append(float(row[0]))
                        y_values.append(float(row[1]))
                    except ValueError:
                        continue

        output_image_path = os.path.join(self.save_dir, f'{q}.png')

        x_values.sort()

        # Сглаживание данных, если флаг smooth установлен
        if self.smooth:
            y_values_smoothed = smooth_data(y_values)
            x_values_smoothed = smooth_data(x_values)
        else:
            y_values_smoothed = y_values
            x_values_smoothed = x_values
        y_values_smoothed = [y / self.divide for y in y_values_smoothed]
        x_values_smoothed = [x / self.divide for x in x_values_smoothed]
        center = sum(y_values_smoothed) / len(y_values_smoothed)
        y_values_smoothed = [center - y for y in y_values_smoothed]
        if draw:
            plt.figure(figsize=(50, 20))
            plt.plot(x_values_smoothed, y_values_smoothed, color='blue', marker='o', linestyle='-', label='График')
            plt.title('')
            plt.xlabel('t(с)')
            plt.ylabel('U(v)')
            plt.grid(True)
            plt.legend()

            plt.savefig(output_image_path)
            print(f"График сохранён как {output_image_path}")

        return math.sqrt(sum([y**2 for y in y_values_smoothed]) / len(y_values_smoothed))

    def get_draw_data(self, draw=True):
        """
        Чтение CSV файлов из директории, генерация графиков и вычисление значений S.
        
        :return: Списки значений S и соответствующих им имен файлов.
        """
        s_values = []
        q_values = []
        csv_files = sorted([f for f in os.listdir(self.csv_dir) if f.endswith('.csv')], key=extract_number)

        for image_file in csv_files:
            csv_file_path = os.path.join(self.csv_dir, image_file)
            file_name = os.path.basename(image_file)
            q = file_name.split('.')[0]
            q_values.append(q)
            s = self.get_s(csv_file_path, q, draw=draw)
            s_values.append(s)

        return s_values, q_values

    def plot_summary_graph(self, s_values, q_values, output_image_name='summary_graph.png', title='График на основе формулы S=√(Σx^2/n) с исключением отсутствующих значений'):
        """
        Построение итогового графика на основе вычисленных значений S.
        Теперь добавляем равномерную шкалу по оси X и исключаем отсутствующие значения.

        :param s_values: Список значений S.
        :param q_values: Список имен файлов (оси X).
        :param output_image_name: Имя файла для сохранения итогового графика.
        """
        # Преобразуем имена файлов в числа
        q_values_numeric = [int(q) for q in q_values]

        # Определяем минимальное и максимальное значение
        min_q = min(q_values_numeric)
        max_q = max(q_values_numeric)

        # Создаем равномерную шкалу по оси X
        uniform_q_values = list(range(min_q, max_q + 1))

        # Создаем списки для точек, исключая отсутствующие значения
        filtered_q_values = []
        filtered_s_values = []

        for q in uniform_q_values:
            if q in q_values_numeric:
                index = q_values_numeric.index(q)
                filtered_q_values.append(q)
                filtered_s_values.append(s_values[index])
            # Пропускаем значения, если данных нет (не добавляем None)

        # Построение графика с соединёнными точками
        plt.figure(figsize=(35, 10))
        plt.plot(filtered_q_values, filtered_s_values, color='blue', marker='o', linestyle='-', label='График', markersize=10)
        plt.title(title)
        plt.xlabel('f')
        plt.ylabel('S')
        plt.grid(True)
        plt.legend()
        plt.xticks(np.arange(0, 255, 5))
        plt.yticks(np.arange(0, 0.2, 0.025))
        output_image_path = os.path.join(self.save_dir, output_image_name)
        plt.savefig(output_image_name)

        print(f"Итоговый график сохранён как {output_image_name}")

    def plot_integrated_graph(self, s_values1, q_values1, s_values2, q_values2,
                          output_image_name='integrated_graph.png', 
                          title='Интегрированный график'):
        """
        Построение интегрированного графика на основе двух наборов значений S.
        
        :param s_values1: Список значений S для первого набора.
        :param q_values1: Список q для первого набора.
        :param s_values2: Список значений S для второго набора.
        :param q_values2: Список q для второго набора.
        :param output_image_name: Имя выходного файла графика.
        :param title: Заголовок графика.
        """
        # Преобразуем q в числовые значения, если они строковые
        q_values1 = [int(q) for q in q_values1]
        q_values2 = [int(q) for q in q_values2]

        min_q = min(q_values1 + q_values2)
        max_q = max(q_values1 + q_values2)

        plt.figure(figsize=(35, 10))
        # Сортировка 1-го графика
        combined1 = sorted(zip(q_values1, s_values1), key=lambda pair: pair[0])
        q_values1_sorted, s_values1_sorted = zip(*combined1)

        # Сортировка 2-го графика
        combined2 = sorted(zip(q_values2, s_values2), key=lambda pair: pair[0])
        q_values2_sorted, s_values2_sorted = zip(*combined2)

        # Построение
        plt.plot(q_values1_sorted, s_values1_sorted, color='blue', marker='o', linestyle='-', label='1mm', markersize=10)
        plt.plot(q_values2_sorted, s_values2_sorted, color='red', marker='o', linestyle='-', label='2mm', markersize=10)

        plt.title(title)
        plt.xlabel('f')
        plt.ylabel('S')
        plt.yticks(np.arange(0, 0.2, 0.025))
        plt.xticks(np.arange(0, 255, 5))
        plt.grid(True)
        plt.legend()
        plt.xlim(min(min(q_values1), min(q_values2)), max(max(q_values1), max(q_values2)))

        output_image_path = os.path.join(self.save_dir, output_image_name)
        plt.savefig(output_image_path)

        print(f"Интегрированный график сохранён как {output_image_path}")

 
if __name__ == "__main__":
    # Создание графика для экспериментальных данных
    # 1mm
    generator_1mm = GraphGenerator(
        csv_dir='D:/PROJECTs/leaves_detection/magnet/Magnet_clean/exp_csv/a1', 
        save_dir='media/a1', 
        smooth=True,
        divide=1000)
    s_values_1mm, q_values_1mm = generator_1mm.get_draw_data(draw=False)
    generator_1mm.plot_summary_graph(s_values_1mm, q_values_1mm, output_image_name='1mm.png', title='1mm')

    # 2mm
    generator_2mm = GraphGenerator(
        csv_dir='D:/PROJECTs/leaves_detection/magnet/Magnet_clean/exp_csv/a2', 
        save_dir='media/a2', 
        smooth=True,
        divide=1000)
    s_values_2mm, q_values_2mm = generator_2mm.get_draw_data(draw=False)
    generator_2mm.plot_summary_graph(s_values_2mm, q_values_2mm, output_image_name='2mm.png', title='2mm')

    # Объединённый график (используем один генератор для построения)
    generator_2mm.plot_integrated_graph(
        s_values1=s_values_1mm,
        q_values1=q_values_1mm,
        s_values2=s_values_2mm,
        q_values2=q_values_2mm,
        output_image_name='integrated_graph.png',
        title='Интегрированный график'
    )

    # # Создание графика для теоритических данных
    # generator = GraphGenerator(
    #     csv_dir='csv/a1', 
    #     save_dir='media/teory_a1', 
    #     smooth=True)
    # s_values, q_values = generator.get_draw_data(draw=False)
    # generator.plot_summary_graph(s_values, q_values, output_image_name='1mm_teory.png')

    # generator = GraphGenerator(
    #     csv_dir='csv/a2', 
    #     save_dir='media/teory_a2', 
    #     smooth=True)
    # s_values, q_values = generator.get_draw_data(draw=False)
    # generator.plot_summary_graph(s_values, q_values, output_image_name='2mm_teory.png')
