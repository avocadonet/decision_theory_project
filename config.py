from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "orig_data"
PROCESSED_DIR = DATA_DIR / "processed_data"
RESULTS_DIR = BASE_DIR / "results"

FIRST_GAME_RESULTS_DIR = RESULTS_DIR / "first_game"
SECOND_GAME_RESULTS_DIR = RESULTS_DIR / "second_game"

N_STUDENTS = 500
M_COURSES = 20
K_COURSES = 5
RANDOM_SEED = 42    

def ensure_directories():
    directories = [
        DATA_DIR,
        GENERATED_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        FIRST_GAME_RESULTS_DIR,
        SECOND_GAME_RESULTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)