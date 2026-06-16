import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка путей для работы из папки src/visualization/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sns.set_theme(style="whitegrid")

def plot_global_poa_comparison():
    """
    Собирает результаты всех экспериментов и строит график динамики PoA.
    Ось X теперь категориальная и подписывает комбинацию всех параметров (N, M, K).
    """
    exp_dir = Path("experiment_results")
    if not exp_dir.exists():
        print("Папка experiment_results не найдена. Сначала запустите run_experiments.py")
        return

    data = []
    
    # Обходим все папки с результатами экспериментов
    for p in exp_dir.glob("exp_N*"):
        summary_file = p / "second_game" / "decentralized_summary.csv"
        if summary_file.exists():
            # Извлекаем все 3 параметра из названия папки (например, exp_N50_M5_K2)
            parts = p.name.split("_")
            try:
                n_val = int(parts[1].replace("N", ""))
                m_val = int(parts[2].replace("M", ""))
                k_val = int(parts[3].replace("K", ""))
                
                # Создаем компактную многострочную подпись для оси X, чтобы текст не сливался
                config_label = f"N={n_val}\nM={m_val}\nK={k_val}"
            except (IndexError, ValueError):
                # Пропускаем, если папка названа некорректно
                continue
            
            df = pd.read_csv(summary_file)
            for _, row in df.iterrows():
                data.append({
                    "N_Students": n_val,           # Используется для правильной сортировки по масштабу
                    "Config_Label": config_label,   # Пойдет на ось X
                    "Mechanism": "Приоритет по рейтингу" if row["mechanism"] == "rating" else "Случайный выбор",
                    "PoA": row["Price_of_Anarchy"]
                })
                
    if not data:
        print("Не удалось найти файлы decentralized_summary.csv для построения глобального графика.")
        return

    # Создаем DataFrame и строго сортируем его по возрастанию масштаба (числа студентов)
    plot_df = pd.DataFrame(data).sort_values("N_Students")

    plt.figure(figsize=(13, 7))
    
    # Строим линейный график. Флаг sort=False критически важен: он заставляет Seaborn
    # выстроить текстовые подписи ровно в том порядке, в котором мы отсортировали DataFrame
    ax = sns.lineplot(
        data=plot_df, 
        x="Config_Label", 
        y="PoA", 
        hue="Mechanism", 
        marker="o", 
        linewidth=2.5, 
        palette=["#C44E52", "#DD8452"],
        sort=False 
    )

    # Добавляем точные числовые подписи над каждой точкой
    for _, row in plot_df.iterrows():
        ax.text(
            row["Config_Label"], 
            row["PoA"] + 0.005, 
            f"{row['PoA']:.3f}", 
            ha="center", 
            va="bottom", 
            fontsize=9, 
            fontweight="bold"
        )

    plt.title("Динамика Цены Анархии (PoA) при различных конфигурациях системы", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Конфигурация параметров эксперимента (Масштаб)", fontsize=11, labelpad=10)
    plt.ylabel("Цена Анархии (Price of Anarchy)", fontsize=11)
    
    # Немного приподнимаем верхний лимит по Y, чтобы текстовые метки над точками не срезались границей
    plt.ylim(0.9, max(plot_df["PoA"]) + 0.04)
    plt.legend(title="Игровой механизм", title_fontsize='11', loc="upper left")
    
    save_path = exp_dir / "global_poa_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"\n[УСПЕХ] Глобальный аналитический график обновлен и сохранен в: {save_path}")

def plot_global_execution_time():
    """Собирает время выполнения из всех экспериментов и строит график сравнения скорости."""
    exp_dir = Path("experiment_results")
    if not exp_dir.exists():
        return

    data = []
    
    for p in exp_dir.glob("exp_N*"):
        dec_summary_file = p / "second_game" / "decentralized_summary.csv"
        cent_summary_file = p / "first_game" / "centralized_objective_value.csv"
        
        if dec_summary_file.exists() and cent_summary_file.exists():
            parts = p.name.split("_")
            try:
                n_val = int(parts[1].replace("N", ""))
                m_val = int(parts[2].replace("M", ""))
                k_val = int(parts[3].replace("K", ""))
                config_label = f"N={n_val}\nM={m_val}\nK={k_val}"
            except (IndexError, ValueError):
                continue
            
            # Централизованное время
            df_cent = pd.read_csv(cent_summary_file)
            if "execution_time_sec" in df_cent.columns:
                data.append({
                    "N_Students": n_val,
                    "Config_Label": config_label,
                    "Algorithm": "ЦЛП Солвер (Оптимум)",
                    "Time_sec": df_cent["execution_time_sec"].iloc[0]
                })
                
            # Децентрализованное время
            df_dec = pd.read_csv(dec_summary_file)
            if "execution_time_sec" in df_dec.columns:
                for _, row in df_dec.iterrows():
                    mech_name = "Best Response (Рейтинг)" if row["mechanism"] == "rating" else "Best Response (Случайно)"
                    data.append({
                        "N_Students": n_val,
                        "Config_Label": config_label,
                        "Algorithm": mech_name,
                        "Time_sec": row["execution_time_sec"]
                    })
                    
    if not data:
        print("Данные о времени выполнения не найдены.")
        return

    plot_df = pd.DataFrame(data).sort_values("N_Students")

    plt.figure(figsize=(13, 7))
    ax = sns.lineplot(
        data=plot_df, 
        x="Config_Label", 
        y="Time_sec", 
        hue="Algorithm", 
        marker="s", 
        linewidth=2.5,
        palette=["#55A868", "#C44E52", "#DD8452"],
        sort=False 
    )

    plt.title("Сравнение времени выполнения алгоритмов (Скорость сходимости)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Конфигурация параметров эксперимента (Масштаб)", fontsize=11, labelpad=10)
    plt.ylabel("Время выполнения (секунды)", fontsize=11)
    
    # Логарифмическая шкала по Y отлично подходит, так как перебор комбинаций растет экспоненциально
    ax.set_yscale("log")
    
    plt.legend(title="Алгоритм", title_fontsize='11', loc="upper left")
    
    save_path = exp_dir / "global_execution_time.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[УСПЕХ] График скорости выполнения сохранен в: {save_path}")

if __name__ == "__main__":
    plot_global_poa_comparison()