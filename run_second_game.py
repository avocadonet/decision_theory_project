import pandas as pd
import numpy as np
from pathlib import Path
from config import PROCESSED_DIR, GENERATED_DIR, SECOND_GAME_RESULTS_DIR, ensure_directories
from models.decentralized_game import run_best_response_dynamics

def load_data():
    meta = pd.read_csv(GENERATED_DIR / "meta.csv")
    utility_df = pd.read_csv(PROCESSED_DIR / "total_utility.csv", index_col="student_id")
    capacities_df = pd.read_csv(GENERATED_DIR / "capacity.csv").sort_values("course_id")
    ratings_df = pd.read_csv(GENERATED_DIR / "student_rating.csv").sort_values("student_id")
    
    k_courses = int(meta.loc[0, "k_courses_per_student"])
    
    return utility_df.values, capacities_df["capacity"].values, ratings_df["rating"].values, k_courses

def run_second_game():
    print("=== SECOND GAME START (DECENTRALIZED) ===")
    ensure_directories()
    
    utility_matrix, capacities, ratings, k_courses = load_data()
    
    # Механизм по рейтингу
    print("\n[1/2] Running Best Response Dynamics (Mechanism: Rating)...")
    strat_rating, loads_rating = run_best_response_dynamics(
        utility_matrix, capacities, ratings, k_courses, mechanism="rating"
    )
    
    # Свободный механизм (рандом)
    print("\n[2/2] Running Best Response Dynamics (Mechanism: Random)...")
    strat_random, loads_random = run_best_response_dynamics(
        utility_matrix, capacities, ratings, k_courses, mechanism="random"
    )
    
    # Сохраняем результаты
    pd.DataFrame(loads_rating, columns=["actual_load"]).to_csv(SECOND_GAME_RESULTS_DIR / "game_loads_rating.csv", index=False)
    pd.DataFrame(loads_random, columns=["actual_load"]).to_csv(SECOND_GAME_RESULTS_DIR / "game_loads_random.csv", index=False)
    
    print("\n=== SECOND GAME FINISHED ===")

if __name__ == "__main__":
    run_second_game()