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

    # Извлечение амплитуды ускорения через RMS (корень из удвоенного среднего квадрата)
    rms_a = np.sqrt(np.mean(a ** 2))
    A_a = rms_a * np.sqrt(2)  # для чистого синуса

    # Перевод в амплитуду смещения
    omega = 2 * np.pi * freq_hz
    A_s = A_a / (omega ** 2)

    return A_s * 1000  # в мм


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
        a -= np.mean(a)

        # Применение FIR фильтра с нулевой фазой (двусторонний)
        # try:
        #     a_f = filtfilt(self.fir_taps, [1.0], a)  # FIR LPF
        # except ValueError:
        #     print("⚠️  Сигнал слишком короткий для filtfilt. Используется односторонний фильтр.")
        #     a_f = lfilter(self.fir_taps, 1, a)

        # # Применение высокочастотного фильтра (убирает дрейф < 1 Гц)
        # b_hp, a_hp = butter(4, 1.0 / (0.5 * self.fs), btype='highpass')
        # a_f = filtfilt(b_hp, a_hp, a_f)

        #         # Обрезка начала и конца по 2 секунды
        # if t[-1] < 4.0:
        #     raise RuntimeError(f"Сигнал слишком короткий для обрезки 2+2 сек: {file_path.name}")
        # t_mask = (t >= 2.0) & (t <= t[-1] - 2.0)
        # t = t[t_mask]
        # a = a[t_mask]
        # a_f = a_f[t_mask]

        # Интеграция и удаление среднего
        v = cumulative_trapezoid(a, t, initial=0)
        s = cumulative_trapezoid(v, t, initial=0)

        # Доп. фильтрация дрейфа в s(t)
        b_drift, a_drift = butter(2, 0.05 / (0.5 * self.fs), btype='highpass')
        s = filtfilt(b_drift, a_drift, s)
        s = detrend(s, type='linear')
        # Расчёт амплитуды
        # amp_mm = np.max(np.abs(s)) 
        f_hz = find_first_int(file_path.stem)
        amp_mm = estimate_amplitude_from_sine(a, t, f_hz)


        return amp_mm


    def run_all(self, summary="displacement_summary.png"):
        labels, res = [], []
        for f in sorted(self.csv_dir.glob("*.csv"),
                        key=lambda p: find_first_int(p.stem)):
            try: 
                res.append(self.process_file(f))
                labels.append(find_first_int(f.stem, f.stem))
            except Exception as e:
                print(f"⚠️  {f.name}: {e}")

        plt.figure(figsize=(12, 5))
        plt.plot(labels, res, "o-", lw=1.8)
        plt.xlabel("F, Гц")
        plt.ylabel("A")
        plt.title("Резонансная кривая шейкера")
        plt.grid(True)
        out = self.save_dir / summary
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.show(); plt.close()
        print(f"Сводный график → {out}")

        with open(self.save_dir / "amplitude_vs_frequency.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frequency_Hz", "M"])
            writer.writerows(zip(labels, res))
        return labels, res

    def plot_single(self, file_path: Path, plot_name="filter_debug.png", show=True):
        # Чтение и подготовка данных
        t, a = self._read_csv(file_path)
        a -= np.mean(a)

        # Применение FIR фильтра с нулевой фазой (двусторонний)
        # try:
        #     a_f = filtfilt(self.fir_taps, [1.0], a)  # FIR LPF
        # except ValueError:
        #     print("⚠️  Сигнал слишком короткий для filtfilt. Используется односторонний фильтр.")
        #     a_f = lfilter(self.fir_taps, 1, a)

        # # Применение высокочастотного фильтра (убирает дрейф < 1 Гц)
        # b_hp, a_hp = butter(4, 1.0 / (0.5 * self.fs), btype='highpass')
        # a_f = filtfilt(b_hp, a_hp, a_f)

                # Обрезка начала и конца по 2 секунды
        # if t[-1] < 4.0:
        #     raise RuntimeError(f"Сигнал слишком короткий для обрезки 2+2 сек: {file_path.name}")
        # t_mask = (t >= 2.0) & (t <= t[-1] - 2.0)
        # t = t[t_mask]
        # a = a[t_mask]
        # a_f = a_f[t_mask]
 
        v = cumulative_trapezoid(a, t, initial=0)
        s = cumulative_trapezoid(v, t, initial=0)

        # Доп. фильтрация дрейфа в s(t)
        b_drift, a_drift = butter(2, 0.1 / (0.5 * self.fs), btype='highpass')
        s = filtfilt(b_drift, a_drift, s)
        s = detrend(s, type='linear')
        
        # Расчёт амплитуды
        amp_mm = np.max(np.abs(s)) * 1_000
        print(f"{file_path.name}: |s|max = {amp_mm:.2f} мм")
        # Визуализация
        fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
        ax[0].plot(t, a, alpha=1, label="raw")
        # ax[0].plot(t, a_f, lw=2, label="filtered")
        ax[0].set_ylabel("a, м/с²"); ax[0].legend(); ax[0].grid()

        ax[1].plot(t, s * 1_000, color="tab:orange", label="s(t), мм")
        ax[1].plot(t, np.abs(s) * 1_000, ls="--", color="tab:red", alpha=.3,
                label="|s|, мм")
        ax[1].set_xlabel("t, c"); ax[1].set_ylabel("s, мм")
        ax[1].set_title(f"|s|max = {amp_mm:.2f} мм"); ax[1].legend(); ax[1].grid()

        # Сохранение
        plt.tight_layout(); out = self.save_dir / plot_name
        plt.savefig(out, dpi=150); plt.close()
        if show:
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



# ────────────────────────────
if __name__ == "__main__":
    analyzer = AccelAmplitudeAnalyzer(
        csv_dir=r"D:\PROJECTs\leaves_detection\magnet\Magnet_clean\exp",
        save_dir=r"D:\PROJECTs\leaves_detection\magnet\Magnet_clean\media",
        cutoff_hz=70.0,
        fs=1_000.0,
        order=512
    )
    analyzer.run_all()
    
    # for f in sorted(analyzer.csv_dir.glob("*.csv"),
    #                     key=lambda p: find_first_int(p.stem)):
    #     print(f"Обработка {f.name}...")
        
    #     analyzer.plot_single(f, plot_name=f.stem + "_filter_debug.png", show=False)

    # analyzer.plot_filter_response(ylim_db=(-150, 5), fname="fir_response.png")