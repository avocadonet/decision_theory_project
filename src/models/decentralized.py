import sys
from pathlib import Path
from itertools import combinations
import time as tm
import numpy as np
import pandas as pd

# Автоматическое добавление корня проекта в пути поиска модулей
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROCESSED_DIR, 
    GENERATED_DIR, 
    SECOND_GAME_RESULTS_DIR, 
    FIRST_GAME_RESULTS_DIR,
    ensure_directories
)


def run_best_response_dynamics(utility_matrix, capacities, ratings, k_courses, mechanism="rating", max_iter=100):
    """Алгоритм Best Response Dynamics для поиска Равновесия Нэша"""
    n_students, m_courses = utility_matrix.shape
    
    current_strategies = np.zeros((n_students, m_courses), dtype=int)
    for i in range(n_students):
        random_courses = np.random.choice(m_courses, k_courses, replace=False)
        current_strategies[i, random_courses] = 1

    all_course_combinations = list(combinations(range(m_courses), k_courses))
    
    for iteration in range(max_iter):
        strategy_changed = False
        demand = current_strategies.sum(axis=0)
         
        for i in range(n_students):
            current_choice = current_strategies[i].copy()
            best_utility = -np.inf
            best_choice = current_choice
            
            demand -= current_choice
            
            for combo in all_course_combinations:
                expected_utility = 0
                combo_strategy = np.zeros(m_courses, dtype=int)
                combo_strategy[list(combo)] = 1
                
                for j in combo:
                    current_course_demand = demand[j] + 1
                    cap = capacities[j]
                     
                    prob_admission = 0.0
                    if current_course_demand <= cap:
                        prob_admission = 1.0
                    else:
                        if mechanism == "random":
                            prob_admission = cap / current_course_demand
                        elif mechanism == "rating":
                            competitors_mask = current_strategies[:, j] == 1
                            competitors_ratings = ratings[competitors_mask]
                            higher_rating_count = np.sum(competitors_ratings > ratings[i])
                            if higher_rating_count < cap:
                                prob_admission = 1.0
                            else:
                                prob_admission = 0.0

                    expected_utility += utility_matrix[i, j] * prob_admission
                
                if expected_utility > best_utility:
                    best_utility = expected_utility
                    best_choice = combo_strategy
            
            demand += best_choice
            
            if not np.array_equal(current_choice, best_choice):
                current_strategies[i] = best_choice
                strategy_changed = True
                
        print(f"Итерация {iteration + 1}: Изменения в стратегиях = {strategy_changed}")
        
        if not strategy_changed:
            print(f"Равновесие Нэша найдено за {iteration + 1} итераций!")
            break
            
    final_demand = current_strategies.sum(axis=0)
    expected_loads = np.minimum(final_demand, capacities)
    
    return current_strategies, expected_loads


def resolve_admissions(strategies, capacities, ratings, mechanism="rating"):
    """Определяет, кто фактически попал на курсы при переполнении."""
    n_students, m_courses = strategies.shape
    actual_assignments = np.zeros_like(strategies)
    
    for j in range(m_courses):
        applicants = np.where(strategies[:, j] == 1)[0]
        demand = len(applicants)
        cap = capacities[j]
        
        if demand <= cap:
            actual_assignments[applicants, j] = 1
        else:
            if mechanism == "rating":
                applicant_ratings = ratings[applicants]
                top_indices = applicants[np.argsort(applicant_ratings)[-cap:]]
                actual_assignments[top_indices, j] = 1
            elif mechanism == "random":
                lucky_indices = np.random.choice(applicants, cap, replace=False)
                actual_assignments[lucky_indices, j] = 1
                
    return actual_assignments


def run_game_core(utility_matrix, capacities, ratings, k_courses, mech):
    """
    ПОДФУНКЦИЯ 1 (ГЛАВНАЯ): Математическое ядро игры.
    Проводит симуляцию Best Response, применяет механизм зачисления 
    и рассчитывает итоговую глобальную полезность.
    """
    start_time = tm.time()
    
    # Поиск профиля Равновесия Нэша
    strategies, _ = run_best_response_dynamics(
        utility_matrix, capacities, ratings, k_courses, mechanism=mech
    )
    execution_time = tm.time() - start_time
    
    # Разрешение конфликтов и фактическое зачисление
    actual_assignments = resolve_admissions(strategies, capacities, ratings, mechanism=mech)
    actual_loads = actual_assignments.sum(axis=0)
    
    # Расчет Z_NE
    total_utility = np.sum(actual_assignments * utility_matrix)
    
    return actual_assignments, actual_loads, total_utility, execution_time


def process_and_save_game_results(results_dir, mech, course_ids, actual_assignments, actual_loads, capacities, total_utility, execution_time):
    """
    ПОДФУНКЦИЯ 2 (ВТОРИЧНАЯ): Сохранение результатов.
    Сохраняет матрицы зачислений конкретного механизма в CSV и выводит логи.
    """
    pd.DataFrame(actual_assignments, columns=course_ids).to_csv(
        results_dir / f"decentralized_assignments_{mech}.csv", index_label="student_id"
    )
    pd.DataFrame({"course_id": course_ids, "actual_load": actual_loads, "capacity": capacities}).to_csv(
        results_dir / f"decentralized_course_loads_{mech}.csv", index=False
    )
    
    print(f"Итоговая полезность (Z_NE): {total_utility:.2f}")
    print(f"Время выполнения: {execution_time:.3f} сек.")


def run_decentralized_game():
    """Основная управляющая функция для децентрализованной модели."""
    print("=== SECOND GAME START (DECENTRALIZED) ===")
    ensure_directories()
    
    # Загрузка метаданных
    meta = pd.read_csv(GENERATED_DIR / "meta.csv")
    k_courses = int(meta.loc[0, "k_courses_per_student"])
    
    utility_df = pd.read_csv(PROCESSED_DIR / "total_utility.csv", index_col="student_id")
    utility_matrix = utility_df.values
    
    capacities_df = pd.read_csv(GENERATED_DIR / "capacity.csv").sort_values("course_id")
    capacities = capacities_df["capacity"].values
    course_ids = capacities_df["course_id"].values
    
    try:
        ratings_df = pd.read_csv(GENERATED_DIR / "student_rating.csv").sort_values("student_id")
        ratings = ratings_df["rating"].values
    except FileNotFoundError:
        print("ВНИМАНИЕ: Файл student_rating.csv не найден. Генерируем случайные рейтинги на лету.")
        n_students = utility_matrix.shape[0]
        ratings = np.random.uniform(0, 1, size=n_students)

    results = []

    # Тестируем все механизмы
    for mech in ["rating", "random"]:
        print(f"\nЗапуск механизма: {mech.upper()}")
        
        # 1. Запуск ядра игры
        actual_assignments, actual_loads, total_utility, execution_time = run_game_core(
            utility_matrix, capacities, ratings, k_courses, mech
        )
        
        # 2. Сохранение результатов шага
        process_and_save_game_results(
            SECOND_GAME_RESULTS_DIR, mech, course_ids, 
            actual_assignments, actual_loads, capacities, total_utility, execution_time
        )
        
        # Добавляем в общую сводку для расчета PoA
        results.append({
            "mechanism": mech,
            "total_utility": total_utility,
            "total_assigned": actual_loads.sum(),
            "execution_time_sec": execution_time
        })

    # Сравнение результатов с ЦЛП (Расчет Цены Анархии)
    try:
        z_opt_df = pd.read_csv(FIRST_GAME_RESULTS_DIR / "centralized_objective_value.csv")
        z_opt = z_opt_df.loc[0, "Z_opt"]
        for res in results:
            res["Z_opt"] = z_opt
            res["Price_of_Anarchy"] = z_opt / res["total_utility"] if res["total_utility"] > 0 else np.inf
    except FileNotFoundError:
        print("\nРезультаты первой игры не найдены. Цена Анархии (PoA) не рассчитана.")

    # Финальное сохранение сводного отчета
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(SECOND_GAME_RESULTS_DIR / "decentralized_summary.csv", index=False)
    
    print("\n=== SECOND GAME FINISHED ===")
    print("Сводка результатов:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_decentralized_game()
    
    
