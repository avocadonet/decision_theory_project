import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    PROCESSED_DIR, 
    GENERATED_DIR, 
    SECOND_GAME_RESULTS_DIR, 
    FIRST_GAME_RESULTS_DIR,
    ensure_directories
)
from models.decentralized_game import run_best_response_dynamics

def resolve_admissions(strategies, capacities, ratings, mechanism="rating"):
    """
    Определяет, кто фактически попал на курсы (разрешает конфликты при переполнении).
    Возвращает матрицу фактических зачислений.
    """
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
                # Берем cap студентов с наивысшим рейтингом
                applicant_ratings = ratings[applicants]
                top_indices = applicants[np.argsort(applicant_ratings)[-cap:]]
                actual_assignments[top_indices, j] = 1
            elif mechanism == "random":
                # Выбираем случайных счастливчиков
                lucky_indices = np.random.choice(applicants, cap, replace=False)
                actual_assignments[lucky_indices, j] = 1
                
    return actual_assignments

def run_decentralized_game():
    print("=== SECOND GAME START (DECENTRALIZED) ===")
    ensure_directories()
    
    # Загрузка данных
    meta = pd.read_csv(GENERATED_DIR / "meta.csv")
    k_courses = int(meta.loc[0, "k_courses_per_student"])
    
    utility_df = pd.read_csv(PROCESSED_DIR / "total_utility.csv", index_col="student_id")
    utility_matrix = utility_df.values
    
    capacities_df = pd.read_csv(GENERATED_DIR / "capacity.csv").sort_values("course_id")
    capacities = capacities_df["capacity"].values
    course_ids = capacities_df["course_id"].values
    
    # Если вы добавили student_rating.csv в генерацию
    try:
        ratings_df = pd.read_csv(GENERATED_DIR / "student_rating.csv").sort_values("student_id")
        ratings = ratings_df["rating"].values
    except FileNotFoundError:
        print("ВНИМАНИЕ: Файл student_rating.csv не найден. Генерируем случайные рейтинги на лету.")
        n_students = utility_matrix.shape[0]
        ratings = np.random.uniform(0, 1, size=n_students)

    results = []

    for mech in ["rating", "random"]:
        print(f"\nЗапуск механизма: {mech.upper()}")
        
        # Поиск Равновесия Нэша
        strategies, expected_loads = run_best_response_dynamics(
            utility_matrix, capacities, ratings, k_courses, mechanism=mech
        )
        
        # Разрешение конфликтов (кто реально попал)
        actual_assignments = resolve_admissions(strategies, capacities, ratings, mechanism=mech)
        actual_loads = actual_assignments.sum(axis=0)
        
        # Расчет итоговой полезности (Z_NE)
        total_utility = np.sum(actual_assignments * utility_matrix)
        
        # Сохранение матриц
        pd.DataFrame(actual_assignments, columns=course_ids).to_csv(
            SECOND_GAME_RESULTS_DIR / f"decentralized_assignments_{mech}.csv", index_label="student_id"
        )
        pd.DataFrame({"course_id": course_ids, "actual_load": actual_loads, "capacity": capacities}).to_csv(
            SECOND_GAME_RESULTS_DIR / f"decentralized_course_loads_{mech}.csv", index=False
        )
        
        results.append({
            "mechanism": mech,
            "total_utility": total_utility,
            "total_assigned": actual_loads.sum()
        })
        
        print(f"Итоговая полезность (Z_NE): {total_utility:.2f}")

    # Сравнение с первой игрой (PoA)
    try:
        z_opt_df = pd.read_csv(FIRST_GAME_RESULTS_DIR / "centralized_objective_value.csv")
        z_opt = z_opt_df.loc[0, "Z_opt"]
        for res in results:
            res["Z_opt"] = z_opt
            res["Price_of_Anarchy"] = z_opt / res["total_utility"] if res["total_utility"] > 0 else np.inf
    except FileNotFoundError:
        print("\nРезультаты первой игры не найдены. PoA не рассчитан.")

    # Сохраняем сводку
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(SECOND_GAME_RESULTS_DIR / "decentralized_summary.csv", index=False)
    
    print("\n=== SECOND GAME FINISHED ===")
    print("Сводка:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_decentralized_game()