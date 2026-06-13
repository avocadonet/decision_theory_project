from pathlib import Path
import sys

# Добавляем корень проекта в пути, чтобы работал импорт из config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import SECOND_GAME_RESULTS_DIR, ensure_directories

sns.set_theme(style="whitegrid")

def _ensure_plots_dir(results_dir):
    """Создает папку plots внутри директории с результатами."""
    plots_dir = Path(results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir

def plot_course_loads(loads_df, mechanism_name, save_dir):
    """Строит график загрузки курсов для конкретного механизма."""
    df = loads_df.sort_values("course_id").copy()
    
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = range(len(df))
    
    # Фон - вместимость (c_max)
    ax.barh(y_pos, df["capacity"], color="#D9D9D9", alpha=0.6, label="Вместимость (c_max)")
    
    # Факт - загрузка
    ax.barh(y_pos, df["actual_load"], color="#4C72B0", alpha=0.9, label="Фактическая загрузка")
    
    for i, row in df.iterrows():
        ax.text(
            row["actual_load"] + 0.5, 
            i, 
            f"{int(row['actual_load'])}/{int(row['capacity'])}", 
            va="center", fontsize=9
        )
        
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["course_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Количество студентов")
    ax.set_ylabel("ID Курса")
    ax.set_title(f"Децентрализованный выбор ({mechanism_name}): Загрузка курсов")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_dir / f"decentralized_loads_{mechanism_name}.png", dpi=220, bbox_inches="tight")
    plt.close()

def plot_utility_comparison(summary_df, save_dir):
    """Строит график сравнения Z_opt и Z_NE (Цена Анархии)."""
    if "Z_opt" not in summary_df.columns:
        print("Нет данных Z_opt для сравнения.")
        return 
        
    fig, ax = plt.subplots(figsize=(9, 6))
    
    labels = ["Централизованная\n(Z_opt)", "Децентр. (Рейтинг)\n(Z_NE)", "Децентр. (Случайно)\n(Z_NE)"]
    z_opt = summary_df["Z_opt"].iloc[0]
    z_rating = summary_df[summary_df["mechanism"] == "rating"]["total_utility"].values[0]
    z_random = summary_df[summary_df["mechanism"] == "random"]["total_utility"].values[0]
    
    values = [z_opt, z_rating, z_random]
    colors = ["#55A868", "#C44E52", "#DD8452"]
    
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    
    # Подписи значений на столбцах
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (max(values)*0.01), 
                f"{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    # Добавляем информацию о PoA внутри столбцов децентрализованных механизмов
    poa_rating = summary_df[summary_df["mechanism"] == "rating"]["Price_of_Anarchy"].values[0]
    poa_random = summary_df[summary_df["mechanism"] == "random"]["Price_of_Anarchy"].values[0]
    
    ax.text(1, z_rating / 2, f"PoA = {poa_rating:.3f}", ha="center", color="white", fontweight="bold", fontsize=10)
    ax.text(2, z_random / 2, f"PoA = {poa_random:.3f}", ha="center", color="white", fontweight="bold", fontsize=10)
    
    ax.set_ylabel("Суммарная полезность (Z)")
    ax.set_title("Сравнение общественного благосостояния (Цена Анархии)")
    
    plt.tight_layout()
    plt.savefig(save_dir / "decentralized_utility_comparison.png", dpi=220, bbox_inches="tight")
    plt.close()

def create_decentralized_visualizations():
    """Основная функция запуска визуализации."""
    ensure_directories()
    results_dir = Path(SECOND_GAME_RESULTS_DIR)
    plots_dir = _ensure_plots_dir(results_dir)
    
    # Визуализация загрузок для обоих механизмов
    for mech in ["rating", "random"]:
        try:
            loads = pd.read_csv(results_dir / f"decentralized_course_loads_{mech}.csv")
            plot_course_loads(loads, mech, plots_dir)
            print(f"Создан график: {plots_dir.name}/decentralized_loads_{mech}.png")
        except FileNotFoundError:
            print(f"Файл decentralized_course_loads_{mech}.csv не найден.")

    # Визуализация сравнения полезностей (PoA)
    try:
        summary = pd.read_csv(results_dir / "decentralized_summary.csv")
        plot_utility_comparison(summary, plots_dir)
        print(f"Создан график: {plots_dir.name}/decentralized_utility_comparison.png")
    except FileNotFoundError:
        print("Файл decentralized_summary.csv не найден. График сравнения не построен.")

if __name__ == "__main__":
    create_decentralized_visualizations()