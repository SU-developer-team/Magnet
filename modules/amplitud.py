# accel_displacement.py ─ FIR (rectangle-window) анализатор
#   • читает CSV с ускорением
#   • LP‑фильтр: FIR 128 taps (boxcar), cut‑off = cutoff_hz
#   • HP‑фильтр: 1 Гц Butterworth 4‑го порядка
#   • двойная интеграция → |s|max (мм)
#   • пакетный режим, одиночный график, АЧХ фильтра

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import firwin, freqz, lfilter, butter, filtfilt
from scipy.signal import detrend


# ────────────────────────────
def find_first_int(string: str, default=np.inf) -> int:
    import re
    m = re.search(r"\d+", string)
    return int(m.group()) if m else default

def fir_lowpass_taps(num_taps: int, cutoff_hz: float, fs: float) -> np.ndarray:
    return firwin(num_taps,
                  cutoff_hz / (0.5 * fs),
                  window=("kaiser", 8.0),  # <-- заменили
                  pass_zero="lowpass")

def fir_bandpass_taps(order: int, low_hz: float, high_hz: float, fs: float):
    return firwin(order, [low_hz, high_hz], pass_zero=False, fs=fs, window=("kaiser", 8.0))


def parse_datetime(ts: str) -> float:
    return datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp()
def estimate_amplitude_from_sine(a: np.ndarray, t: np.ndarray, freq_hz: float) -> float:
    # Удаление DC-компоненты
    a = a - np.mean(a)

    # Амплитуда ускорения по максимуму (модуль)
    A_a = np.max(np.abs(a))

    # Перевод в амплитуду смещения по формуле
    omega = 2 * np.pi * freq_hz
    A_s = A_a / (omega ** 2)

    return A_s * 1000 # в метрах



# ────────────────────────────
class AccelAmplitudeAnalyzer:
    def __init__(self,
                 csv_dir: str | Path,
                 save_dir: str | Path,
                 cutoff_hz: float = 70.0,
                 fs: float = 1_000.0,
                 order: int = 128,
                 acc_label: str = "Accelerometer",
                 time_label: str = "Time(s)"):
        self.csv_dir = Path(csv_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.cutoff_hz = float(cutoff_hz)
        self.fs = float(fs)
        self.order = int(order)
        self.acc_label = acc_label
        self.time_label = time_label
        # self.fir_taps = fir_lowpass_taps(self.order, self.cutoff_hz, self.fs)
        self.fir_taps = fir_bandpass_taps(order=512, low_hz=1.0, high_hz=self.cutoff_hz, fs=self.fs)

    def _read_csv(self, file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        t, a = [], []
        with file_path.open(encoding="utf-8") as f:
            rdr = csv.reader(f, delimiter=";")
            header = next(rdr)
            try:
                ti = header.index(self.time_label)
                ai = header.index(self.acc_label)
            except ValueError as e:
                raise ValueError(f"Нет столбцов в {file_path.name}") from e

            for row in rdr:
                try:
                    t.append(parse_datetime(row[ti]))
                    a.append(float(row[ai].replace(",", ".")))
                except Exception as e:
                    print(f"⚠️ Ошибка чтения строки {row} в {file_path.name}: {e}")
                    pass
        if len(a) < 10:
            print(f'A {a}')
            print(f't {t}')

            raise RuntimeError(f"Мало данных в {file_path.name}")
        return np.array(t) - t[0], np.array(a)

    def process_file(self, file_path: Path) -> float:
                # Чтение и подготовка данных
        t, a = self._read_csv(file_path) 
        # amp_mm = np.max(np.abs(s)) 
        f_hz = find_first_int(file_path.stem)
        amp_mm = estimate_amplitude_from_sine(a, t, f_hz)

        # a_dc   = a - np.mean(a)            # убираем DC
        A_a    = np.mean(np.abs(a))      # [м/с²] 

        return amp_mm, A_a


    def run_all(self, summary="displacement_summary.png"):
        freqs, disp_mm, acc_peak = [], [], []

        # ── обходим все файлы ─────────────────────────────
        for f in sorted(self.csv_dir.glob("*.csv"),
                        key=lambda p: find_first_int(p.stem)):
            try:
                d_mm, a_pk = self.process_file(f)
                freqs.append(find_first_int(f.stem, f.stem))
                disp_mm.append(d_mm)
                acc_peak.append(a_pk)
            except Exception as e:
                print(f"⚠️ {f.name}: {e}")

        # ── строим два графика один под другим ────────────
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

        ax1.plot(freqs, disp_mm, "o-")
        ax1.set_ylabel("A, мм")
        ax1.set_title("Амплитуда смещения")

        ax2.plot(freqs, acc_peak, "o-")
        ax2.set_xlabel("F, Гц")
        ax2.set_ylabel("A, м/с²")
        ax2.set_title("Пиковое ускорение")

        for ax in (ax1, ax2):
            ax.grid(True)

        plt.tight_layout()
        out = self.save_dir / summary
        plt.savefig(out, dpi=150)
        plt.show()
        plt.close()
        print(f"Сводный график → {out}")

        # ── экспорт CSV c двумя метриками ─────────────────
        with open(self.save_dir / "amplitude_vs_frequency.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["Frequency_Hz", "Disp_mm", "Acc_peak_mps2"])
            wr.writerows(zip(freqs, disp_mm, acc_peak))

        return freqs, disp_mm, acc_peak


    def plot_single(self, file_path: Path, plot_name="filter_debug.png", show=True):
        # Чтение и подготовка данных
        t, a = self._read_csv(file_path)
        a -= np.mean(a)

        # Расчёт амплитуды по формуле A_s = A_a / (2πf)^2
        f_hz = find_first_int(file_path.stem)
        amp_mm = estimate_amplitude_from_sine(a, t, f_hz)

        print(f"{file_path.name}: A_s = {amp_mm:.2f} мм (по формуле)")

        # Визуализация
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(t, a, alpha=1, label="a(t)")
        ax.set_xlabel("t, с")
        ax.set_ylabel("a(t), м/с²")
        ax.set_title(f"A_s = {amp_mm:.2f} мм @ {f_hz} Гц")
        ax.legend(); ax.grid()

        # Сохранение
        plt.tight_layout()
        out = self.save_dir / plot_name
        plt.savefig(out, dpi=150)
        plt.close()
        plt.show()



    def plot_filter_response(self, ylim_db=(-80, 5), fname="fir_response.png"):
        w, h = freqz(self.fir_taps, worN=4096, fs=self.fs)
        mag_db = 20 * np.log10(np.abs(h) + 1e-12)
        plt.figure(figsize=(9, 4))
        plt.plot(w, mag_db, lw=2, label="FIR 128 taps (rect)")
        plt.axvline(self.cutoff_hz, ls="--", color="crimson", label=f"fc = {self.cutoff_hz} Гц")
        plt.axhline(-3, color="gray", ls=":"); plt.text(self.cutoff_hz * 1.02, -2.5, "-3 dB", va="center")
        plt.xlim(0, 0.5 * self.fs); plt.ylim(*ylim_db)
        plt.xlabel("Частота, Гц"); plt.ylabel("Амплитуда, dB")
        plt.title("АЧХ FIR‑LPF (прямоугольное окно)")
        plt.grid(ls=":"); plt.legend()
        out = self.save_dir / fname
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.show(); plt.close()
        print(f"АЧХ → {out}")

    def plot_displacement_from_formula(self, file_path: Path, plot_name="s_from_formula.png", show=True):
        # Чтение и подготовка данных
        t, a = self._read_csv(file_path)
        # a -= np.mean(a)

        # Извлечение частоты из имени файла
        f_hz = find_first_int(file_path.stem)
        omega_sq = (2 * np.pi * f_hz) ** 2

        # Расчёт смещения по формуле
        s = a / omega_sq  # [м]
        s_mm = s * 1000  # → [мм]

        # График
        plt.figure(figsize=(14, 5))
        plt.plot(t, s_mm, label=f"s(t) [по формуле], {f_hz} Гц", color="tab:red")
        plt.plot(t, a, label="a(t) [ускорение]", alpha=0.5, color="tab:blue")
        plt.xlabel("t, с")
        plt.ylabel("s(t), мм")
        plt.title(f"Смещение по формуле: s(t) = a(t) / (2πf)²")
        plt.grid(True)
        plt.legend()

        # Сохранение
        out = self.save_dir / plot_name
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        if show:
            plt.show()
        plt.close()


# ────────────────────────────
if __name__ == "__main__":
    analyzer = AccelAmplitudeAnalyzer(
        csv_dir=r"D:\PROJECTs\leaves_detection\magnet\Magnet_clean\exp\3",
        save_dir=r"D:\PROJECTs\leaves_detection\magnet\Magnet_clean\media",
        cutoff_hz=70.0,
        fs=1_000.0,
        order=512
    )
    analyzer.run_all()
    # analyzer.plot_displacement_from_formula(Path(r"D:\PROJECTs\leaves_detection\magnet\Magnet_clean\exp\2.csv"))
