from pathlib import Path
import numpy as np
import pandas as pd


def run_greedy_assignment(
    sorted_utility_file: str = "data/total_utility_sorted.csv",
    meta_file: str = "data/meta.csv",
    capacity_file: str = "data/capacity.csv",
    output_dir: str = "data/greedy_assignment_results"
):
    sorted_df = pd.read_csv(sorted_utility_file)
    meta_df = pd.read_csv(meta_file)
    capacity_df = pd.read_csv(capacity_file)

    n = int(meta_df.loc[0, "n_students"])
    m = int(meta_df.loc[0, "m_courses"])
    k = int(meta_df.loc[0, "k_courses_per_student"])

    capacities = capacity_df.sort_values("course_id")["capacity"].to_numpy(dtype=int)

    student_loads = np.zeros(n, dtype=int)
    course_loads = np.zeros(m, dtype=int)
    assignment_matrix = np.zeros((n, m), dtype=int)

    selected_pairs = []

    for _, row in sorted_df.iterrows():
        i = int(row["i"])
        j = int(row["j"])
        u = float(row["u"])

        if student_loads[i] >= k:
            continue

        if course_loads[j] >= capacities[j]:
            continue

        if assignment_matrix[i, j] == 1:
            continue

        assignment_matrix[i, j] = 1
        student_loads[i] += 1
        course_loads[j] += 1

        selected_pairs.append({
            "u": u,
            "i": i,
            "j": j
        })

        if np.all(student_loads >= k):
            break

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_df = pd.DataFrame(selected_pairs)
    selected_df.to_csv(output_path / "greedy_selected_pairs.csv", index=False)

    assignment_df = pd.DataFrame(
        assignment_matrix,
        columns=[f"course_{j}" for j in range(m)]
    )
    assignment_df.insert(0, "student_id", np.arange(n))
    assignment_df.to_csv(output_path / "greedy_assignment_matrix.csv", index=False)

    student_loads_df = pd.DataFrame({
        "student_id": np.arange(n),
        "assigned_courses_count": student_loads
    })
    student_loads_df.to_csv(output_path / "greedy_student_loads.csv", index=False)

    course_loads_df = pd.DataFrame({
        "course_id": np.arange(m),
        "assigned_students_count": course_loads,
        "capacity": capacities
    })
    course_loads_df.to_csv(output_path / "greedy_course_loads.csv", index=False)

    print("Greedy assignment completed.")
    print(f"Students fully assigned: {(student_loads >= k).sum()} / {n}")
    print("\nStudent loads:")
    print(student_loads)
    print("\nCourse loads:")
    print(course_loads)

    return {
        "student_loads": student_loads,
        "course_loads": course_loads,
        "assignment_matrix": assignment_matrix,
        "selected_pairs": selected_pairs,
    }


if __name__ == "__main__":
    run_greedy_assignment()