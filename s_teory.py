# magnet_simulation_refactored.py (v7)
"""
Симуляция + RMS‑метрики внешней и самоиндукционной ЭДС
для *дробных* частот шейкера (в т. ч. 0.5 Гц шаг).

v7 — поддержка float‑частот:
• `batch_run()` принимает `List[float]`.
• Частоты в примере `main()` задаются через `np.arange(start, stop, step)`.
• Форматирование логов/ось X выводят одно‑ или двухзнаковые дроби.
"""
from __future__ import annotations

import csv, logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import pi

from models import Magnet, Shaker, Coil

###############################################################################
# ЛОГГЕР
###############################################################################

def configure_logger(name: str = "magnet_simulation", log_dir: str | Path = "logs",
                     level_file: int = logging.DEBUG, level_console: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(min(level_file, level_console))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fh = logging.FileHandler(Path(log_dir) / f"{ts}.log"); fh.setLevel(level_file); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setLevel(level_console); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

###############################################################################
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
###############################################################################

def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr))))

###############################################################################
# Физика / ОДУ / Симуляция (не изменялась)
###############################################################################

def get_magnet_position(shaker: Shaker, t: float) -> float:
    return shaker.X0 * np.sin(shaker.W * t)

def calculate_f_damping(v: float, magnet: Magnet) -> float:
    Cd, rho = 1.2, 1.225
    area = pi * (magnet.diameter * 0.5) ** 2
    return 0.5 * rho * area * Cd * v ** 2 * np.sign(v)

def calculate_f_air(v: float, magnet: Magnet, gap: float) -> float:
    if gap <= 0:
        raise ValueError("Диаметр магнита должен быть меньше диаметра цилиндра.")
    mu = 1.81e-5
    return -6 * pi * mu * magnet.height * v / gap

def combined_equations(t: float, y: np.ndarray, magnet: Magnet, shaker: Shaker,
                       z_top: float, z_bottom: float, coil: Coil, R: float, gap: float):
    z_m, v_m, z_tm, v_tm, z_bm, v_bm, z_sk, v_sk, i = y
    Fg = magnet.mass * shaker.G
    F_sh = shaker.get_force(magnet, t)
    a_sk = F_sh / magnet.mass
    F_top = magnet.get_force(abs(z_m - magnet.height*0.5 - z_top) + get_magnet_position(shaker, t))
    F_bot = magnet.get_force(abs(z_m - magnet.height*0.5 - z_bottom) + get_magnet_position(shaker, t))
    F_total = -F_top + F_bot - Fg 
    a_m = F_total / magnet.mass
    a_tm = a_bm = a_sk
    _, emf = coil.get_total_emf(shaker, z_m, v_m, t, a_m)
    L = coil.calculate_inductance()
    di_dt = (emf - R * i) / L
    return [v_m, a_m, v_tm, a_tm, v_bm, a_bm, v_sk, a_sk, di_dt]

def run_simulation(magnet: Magnet, shaker: Shaker, coil: Coil, *,
                   z_top: float, z_bottom: float, start_z: float, gap: float,
                   R: float = 0.1, T: float = 5.0, N: int = 5000) -> Dict[str, Any]:
    y0 = [start_z, 0, z_top, 0, z_bottom, 0, 0.0, 0, 0]
    t_eval = np.linspace(0, T, N)
    sol = solve_ivp(combined_equations, (0, T), y0,
                    args=(magnet, shaker, z_top, z_bottom, coil, R, gap),
                    t_eval=t_eval, rtol=1e-6, atol=1e-6)
    L = coil.calculate_inductance()
    v_m = sol.y[1]
    emf_ext = np.array([coil.get_total_emf(shaker, z, v, t, np.nan)[1]
                        for t, z, v in zip(sol.t, sol.y[0], v_m)])
    emf_self = -L * np.gradient(sol.y[8], sol.t)
    return {"eds_external": emf_ext, "eds_self": emf_self, "eds_total": emf_ext + emf_self}

###############################################################################
# Batch‑run S(f)
###############################################################################

def batch_run(freqs: List[float], out_dir: str | Path = "batch_output") -> None:
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    log = logging.getLogger("magnet_simulation")

    magnet = Magnet(diameter=0.0195, mass=0.043, height=0.01)
    z_top, z_bottom = 0.09, 0.01
    gap = (0.0205 - magnet.diameter) / 2
    coil = Coil(turns_count=208, thickness=0.01025, radius=0.01025, position=0.015,
                magnet=magnet, wire_diameter=0.000961538462, layer_count=4)

    table: List[Tuple[float, float, float, float]] = []
    for f in freqs:
        shaker = Shaker(G=9.8, miew=f, X0=0.001)
        sim = run_simulation(magnet, shaker, coil, z_top=z_top, z_bottom=z_bottom,
                             start_z=0.0425, gap=gap)
        s_ext, s_self, s_tot = rms(sim['eds_external']), rms(sim['eds_self']), rms(sim['eds_total'])
        table.append((f, s_ext, s_self, s_tot))
        log.info(f"f={f:5.2f} Hz | S_ext={s_ext:.3e} | S_self={s_self:.3e} | S_tot={s_tot:.3e}")

    # CSV
    csv_path = out_dir / "s_vs_freq.csv"
    with open(csv_path, "w", newline="") as fcsv:
        csv.writer(fcsv).writerows([["freq_Hz","S_ext","S_self","S_total"]]+table)
    log.info(f"CSV saved to {csv_path}")

    # Combined plot
    f_arr, s_ext_arr, s_self_arr, _ = zip(*table)
    plt.figure(figsize=(12,6))
    plt.plot(f_arr, s_ext_arr, marker='o', linestyle='-', label='External EMF')
    plt.plot(f_arr, s_self_arr, marker='s', linestyle='--', label='Self‑induction EMF')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RMS EMF (V)')
    plt.title('RMS of External vs Self‑induction EMF')
    plt.xticks(f_arr, [f"{x:.2g}" for x in f_arr])
    plt.grid(True); plt.legend()
    img_path = out_dir / "s_ext_self_combined.png"
    plt.tight_layout(); plt.savefig(img_path); plt.show()
    log.info(f"Plot saved to {img_path}")

###############################################################################
# MAIN
###############################################################################

def main():
    configure_logger()
    # Frequencies from 1 Hz to 10 Hz with 0.5 Hz step
    freqs = list(np.arange(1.0, 15.1, 0.5))  # [1.0, 1.5, …, 10.0]
    batch_run(freqs)

if __name__ == '__main__':
    main()
