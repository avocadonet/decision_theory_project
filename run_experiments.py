import shutil
import importlib
import re
from pathlib import Path

# Синхронизация путей с новой архитектурой проекта
import config
import src.data_prep.generate_raw as gen_data
import src.data_prep.build_utility as tot_util
import src.models.centralized as cent_opt
import src.visualization.plot_centralized as cent_vis
import src.models.decentralized as dec_game
import src.visualization.plot_decentralized as dec_vis
import src.visualization.plot_global_comparison as global_vis

# Полный набор из 10 конфигураций для комплексного исследования моделей
EXPERIMENTS = [
    {"N_STUDENTS": 20, "M_COURSES": 4, "K_COURSES": 1},   # 1. Микро-кейс (проверка сходимости)
    {"N_STUDENTS": 50, "M_COURSES": 5, "K_COURSES": 2},   # 2. Исходный малый кейс
    {"N_STUDENTS": 80, "M_COURSES": 6, "K_COURSES": 2},   # 3. Легкий дефицит мест
    {"N_STUDENTS": 100, "M_COURSES": 8, "K_COURSES": 3},  # 4. Средний масштаб
    {"N_STUDENTS": 150, "M_COURSES": 10, "K_COURSES": 3}, # 5. Исходный средний кейс
    {"N_STUDENTS": 200, "M_COURSES": 12, "K_COURSES": 3}, # 6. Рост конкуренции на стабильные курсы
    {"N_STUDENTS": 250, "M_COURSES": 12, "K_COURSES": 4}, # 7. Плотный поток, дефицит слотов
    {"N_STUDENTS": 300, "M_COURSES": 15, "K_COURSES": 4}, # 8. Исходный крупный кейс
    {"N_STUDENTS": 400, "M_COURSES": 18, "K_COURSES": 4}, # 9. Высокая нагрузка на систему
    {"N_STUDENTS": 500, "M_COURSES": 20, "K_COURSES": 5}, # 10. Экстремальный стресс-тест для алгоритма Best Response
]

def update_config_on_disk(n, m, k):
    """Обновляет параметры в файле config.py с помощью регулярных выражений."""
    with open("config.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    content = re.sub(r"N_STUDENTS\s*=\s*\d+", f"N_STUDENTS = {n}", content)
    content = re.sub(r"M_COURSES\s*=\s*\d+", f"M_COURSES = {m}", content)
    content = re.sub(r"K_COURSES\s*=\s*\d+", f"K_COURSES = {k}", content)
    
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(content)

def run_pipeline(n, m, k):
    """Обновляет кэш оперативной памяти и запускает все модули симуляции поочередно."""
    update_config_on_disk(n, m, k)
    
    # Сбрасываем кэш импортов, заставляя модули прочитать свежие значения N, M, K из диска
    importlib.reload(config)
    importlib.reload(gen_data)
    importlib.reload(tot_util)
    importlib.reload(cent_opt)
    importlib.reload(cent_vis)
    importlib.reload(dec_game)
    importlib.reload(dec_vis)

    print(f"\n\n{'='*60}")
    print(f"=== ЗАПУСК СЦЕНАРИЯ СИМУЛЯЦИИ: N={n}, M={m}, K={k} ===")
    print(f"{'='*60}\n")
    
    # Шаг 1: Генерация сырых данных о курсах и студентах
    gen_data.generate_initial_data()
    
    # Шаг 2: Расчет и сохранение полной матрицы полезностей
    tot_util.generate_total_utility_matrix(
        data_dir=config.GENERATED_DIR, 
        output_file=config.PROCESSED_DIR / "total_utility.csv"
    )
    print("[ДАННЫЕ] Матрица полезности (total_utility.csv) успешно пересчитана.")
    
    # Шаг 3: Поиск централизованного оптимума (ЦЛП) и его графики
    cent_opt.solve_centralized_optimization()
    cent_vis.create_all_visualizations()
    
    # Шаг 4: Поиск децентрализованного Равновесия Нэша (Игра) и его графики
    dec_game.run_decentralized_game()
    dec_vis.create_decentralized_visualizations()

def backup_results(n, m, k):
    """Архивирует результаты текущей итерации в папку с кодированием всех трех параметров."""
    backup_dir = Path(f"experiment_results/exp_N{n}_M{m}_K{k}")
    
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
        
    shutil.copytree("results", backup_dir)
    print(f"\n[АРХИВ] Данные и графики шага сохранены в: {backup_dir}")

if __name__ == "__main__":
    # Инициализация корневой папки для долгосрочного хранения результатов экспериментов
    Path("experiment_results").mkdir(exist_ok=True)
    
    # Проход по всему списку из 10 конфигураций
    for idx, exp in enumerate(EXPERIMENTS, 1):
        print(f"\nПрогресс пайплайна: Сценарий {idx} из {len(EXPERIMENTS)}")
        run_pipeline(exp["N_STUDENTS"], exp["M_COURSES"], exp["K_COURSES"])
        backup_results(exp["N_STUDENTS"], exp["M_COURSES"], exp["K_COURSES"])
        
    # Финальный шаг — сборка всех данных в один большой аналитический график тренда
    print("\nФормирование глобальной аналитики по всем 10 проведенным экспериментам...")
    importlib.reload(global_vis)
    global_vis.plot_global_poa_comparison()
    global_vis.plot_global_execution_time()
    print("\n[ЗАВЕРШЕНО] Все 10 конфигураций успешно просчитаны. Итоговый график-тренд собран!")