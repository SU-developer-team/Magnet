import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import os
from scipy.ndimage import uniform_filter1d
import re


def extract_number(file_name):
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]) if numbers else float('inf')


def smooth_data(data, window_size=5):
    return uniform_filter1d(data, size=window_size)


class CoilGraphGenerator:
    def __init__(self, csv_dir, save_dir, smooth=True, divide=1):
        self.csv_dir = csv_dir
        self.save_dir = save_dir
        self.smooth = smooth
        self.divide = divide

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def get_s_from_coil(self, csv_file_path, q, draw=True):
        time_values = []
        coil_values = []

        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';')
            header = next(csvreader)  # Считываем заголовок
            time_idx = header.index("Time(s)")
            coil_idx = header.index("Coil")

            for row in csvreader:
                try:
                    time = row[time_idx]
                    coil = float(row[coil_idx].replace(',', '.'))
                    timestamp_parts = time.split()[-1].split(':')  # только время
                    t = float(timestamp_parts[2])  # только миллисекунды
                    time_values.append(t)
                    coil_values.append(coil)
                except:
                    continue

        output_image_path = os.path.join(self.save_dir, f'{q}.png')

        if self.smooth:
            y_values_smoothed = smooth_data(coil_values)
            x_values_smoothed = smooth_data(time_values)
        else:
            y_values_smoothed = coil_values
            x_values_smoothed = time_values

        y_values_smoothed = [y / self.divide for y in y_values_smoothed]
        x_values_smoothed = [x / self.divide for x in x_values_smoothed]

        center = sum(y_values_smoothed) / len(y_values_smoothed)
        y_values_centered = [center - y for y in y_values_smoothed]

        if draw:
            plt.figure(figsize=(20, 8))
            plt.plot(x_values_smoothed, y_values_centered, color='blue', linestyle='-', label='Coil')
            plt.title(f'Coil data: {q}')
            plt.xlabel('Time (ms)')
            plt.ylabel('Coil (V)')
            plt.grid(True)
            plt.legend()
            plt.savefig(output_image_path)
            plt.close()
            print(f"График сохранён: {output_image_path}")

        return math.sqrt(sum(y**2 for y in y_values_centered) / len(y_values_centered))

    def generate_all_graphs(self, draw=True):
        s_values = []
        q_values = []
        csv_files = sorted([f for f in os.listdir(self.csv_dir) if f.endswith('.csv')], key=extract_number)

        for file in csv_files:
            file_path = os.path.join(self.csv_dir, file)
            file_name = os.path.splitext(file)[0]
            q_values.append(file_name)
            s = self.get_s_from_coil(file_path, file_name, draw=draw)
            s_values.append(s)

        return s_values, q_values

    def plot_summary_s_graph(self, s_values, q_values, output_name='summary_s.png'):
        q_numeric = [int(q) for q in q_values]
        plt.figure(figsize=(20, 8))
        plt.plot(q_numeric, s_values, marker='o', linestyle='-', label='S')
        plt.title('S = √(Σ(Coil²)/n)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('S')
        plt.grid(True)
        plt.legend()
        plt.xticks(np.arange(2, 71, 1))
        # plt.yticks(np.arange(0, 0.0006, 0.0001))
        output_path = os.path.join(self.save_dir, output_name)
        plt.savefig(output_path)
        plt.close()
        print(f"Сводный график сохранён: {output_path}")


# Пример использования
if __name__ == "__main__":
    generator = CoilGraphGenerator(
        csv_dir=r'D:\shaker\A3_DUBLE_MAGNETS_V2',   # ← УКАЖИ ПУТЬ К ПАПКЕ С CSV
        save_dir=r'D:\shaker\A3_DUBLE_MAGNETS_V2', # ← УКАЖИ ПАПКУ ДЛЯ ГРАФИКОВ
        smooth=False,         # ← ОТКЛЮЧИТЬ СГЛАЖИВАНИЕ
        divide=1_000               # ← Масштабировать на 1000
    )
    s_values, q_values = generator.generate_all_graphs(draw=False)
    generator.plot_summary_s_graph(s_values, q_values, output_name='coil_summary.png')
