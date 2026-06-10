from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import (
    GENERATED_DIR,
    N_STUDENTS,
    M_COURSES,
    K_COURSES,
    RANDOM_SEED,
    ensure_directories,
)


def generate_initial_data():
    ensure_directories()

    n = N_STUDENTS
    m = M_COURSES
    k = K_COURSES
    seed = RANDOM_SEED
    output_path = Path(GENERATED_DIR)

    rng = np.random.default_rng(seed)

    student_ids = np.arange(n)
    course_ids = np.arange(m)

    usefulness_rows = []
    for i in student_ids:
        for j in course_ids:
            usefulness_rows.append({
                "student_id": int(i),
                "course_id": int(j),
                "usefulness_course": round(float(rng.uniform(0, 1)), 4)
            })

    pd.DataFrame(usefulness_rows).to_csv(
        output_path / "usefulness_course.csv",
        index=False
    )

    pd.DataFrame({
        "course_id": course_ids,
        "quality_teaching": np.round(rng.uniform(5, 10, size=m), 4)
    }).to_csv(output_path / "quality_teaching.csv", index=False)

    pd.DataFrame({
        "course_id": course_ids,
        "ease_passing_exam": np.round(rng.uniform(3, 9, size=m), 4)
    }).to_csv(output_path / "ease_passing_exam.csv", index=False)

    pd.DataFrame({
        "student_id": student_ids,
        "student_laziness": np.round(rng.uniform(0, 1, size=n), 4)
    }).to_csv(output_path / "student_laziness.csv", index=False)

    pd.DataFrame({
        "student_id": student_ids,
        "importance_certainty": np.round(rng.uniform(0, 1, size=n), 4)
    }).to_csv(output_path / "importance_certainty.csv", index=False)

    uncertainty = rng.choice([0.5, 2.0], size=m, p=[0.75, 0.25])
    pd.DataFrame({
        "course_id": course_ids,
        "exchange_rate_uncertainty": np.round(uncertainty, 4)
    }).to_csv(output_path / "exchange_rate_uncertainty.csv", index=False)

    pd.DataFrame({
        "course_id": course_ids,
        "capacity": rng.choice([25, 30, 35], size=m)
    }).to_csv(output_path / "capacity.csv", index=False)

    pd.DataFrame({
        "course_id": course_ids,
        "min_quantity": np.full(m, 15)
    }).to_csv(output_path / "min_quantity.csv", index=False)

    pd.DataFrame([{
        "n_students": n,
        "m_courses": m,
        "k_courses_per_student": k,
        "random_seed": seed
    }]).to_csv(output_path / "meta.csv", index=False)

    # 9) student_certainty_weight: вектор n
    pd.DataFrame({
        "student_id": student_ids,
        "student_certainty_weight": np.round(rng.uniform(0, 1, size=n), 4)
    }).to_csv(output_path / "student_certainty_weight.csv", index=False)
    
    print(f"Data generated in: {output_path.resolve()}")
    print("Created files:")
    for file in sorted(output_path.glob("*.csv")):
        print(f" - {file.name}")


if __name__ == "__main__":
    generate_initial_data()