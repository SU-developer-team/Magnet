# calibrate_acc.py
from pathlib import Path
import csv

# ─── Настройки ────────────────────────────────────────────────────────────────
INPUT_DIR   = Path(r"D:\shaker\A3_DUBLE_MAGNETS_V2")               # ваша директория с исходными CSV
OUTPUT_DIR  = Path(r"D:\PROJECTs\leaves_detection\magnet\Magnet_clean\exp\calibrated_2")    # куда сохранить откалиброванные файлы
K_ACCEL     = 0.054999985         # коэффициент x¹ для канала CH2 (Accelerometer)

OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Обработка всех CSV ───────────────────────────────────────────────────────
for csv_path in INPUT_DIR.glob("*.csv"):
    # пропускаем вспомогательные файлы/папки
    if csv_path.name.startswith("_"):  
        continue

    with csv_path.open(encoding="utf-8") as fin, \
         (OUTPUT_DIR / csv_path.name).open("w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin, delimiter=';')
        writer = csv.writer(fout, delimiter=';')

        header = next(reader)
        writer.writerow(header)

        acc_idx = header.index("Accelerometer")   # позиция столбца

        for row in reader:
            try:
                # преобразуем «запятую» в «точку», множим, возвращаем запятую
                raw_val = float(row[acc_idx].replace(',', '.'))
                new_val = raw_val * K_ACCEL
                row[acc_idx] = f"{new_val:.3f}".replace('.', ',')
            except (ValueError, IndexError):
                # если строка неполная / не-число ‒ пишем как есть
                pass

            writer.writerow(row)

print("Готово: исправленные файлы находятся в папке _out/")
