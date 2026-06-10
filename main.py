from models.total_utility import generate_total_utility_matrix
from models.sort_total_utility import sort_total_utility
from models.greedy_assignment import run_greedy_assignment

from config import (
    GENERATED_DIR,
    N_STUDENTS,
    M_COURSES,
    K_COURSES,
    RANDOM_SEED,
    ensure_directories,
    PROCESSED_DIR,
)

def run_first_game():
    print("=== FIRST GAME START ===")

    ensure_directories()

    print("\n[1/3] Generating total utility matrix...")
    generate_total_utility_matrix(data_dir=GENERATED_DIR, output_file=PROCESSED_DIR / "total_utility.csv", )

    print("\n[2/3] Sorting total utility...")
    sort_total_utility()

    print("\n[3/3] Running greedy assignment...")
    run_greedy_assignment()

    print("\n=== FIRST GAME FINISHED ===")


if __name__ == "__main__":
    run_first_game()