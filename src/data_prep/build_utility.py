from pathlib import Path
import pandas as pd


def calculate_course_utility(
    beta_i: float,
    w_ij: float,
    a_j: float,
    d_j: float,
    p_i: float,
    lambda_value: float,
    sigma_j: float
) -> float:
    return (
        beta_i * w_ij * a_j
        + (1 - beta_i) * w_ij * d_j
        - p_i * lambda_value * sigma_j
    )


def generate_total_utility_matrix(
    data_dir: str = "data",
    output_file: str = "data/total_utility.csv",
):
    data_path = Path(data_dir)

    usefulness_course = pd.read_csv(data_path / "usefulness_course.csv")
    quality_teaching = pd.read_csv(data_path / "quality_teaching.csv")
    ease_passing_exam = pd.read_csv(data_path / "ease_passing_exam.csv")
    student_laziness = pd.read_csv(data_path / "student_laziness.csv")
    importance_certainty = pd.read_csv(data_path / "importance_certainty.csv")
    exchange_rate_uncertainty = pd.read_csv(data_path / "exchange_rate_uncertainty.csv")
    student_certainty_weight = pd.read_csv(data_path / "student_certainty_weight.csv")

    df = usefulness_course.merge(quality_teaching, on="course_id", how="left")
    df = df.merge(ease_passing_exam, on="course_id", how="left")
    df = df.merge(student_laziness, on="student_id", how="left")
    df = df.merge(importance_certainty, on="student_id", how="left")
    df = df.merge(exchange_rate_uncertainty, on="course_id", how="left")
    df = df.merge(student_certainty_weight, on="student_id", how="left")

    df["total_utility"] = df.apply(
        lambda row: calculate_course_utility(
            beta_i=row["student_laziness"],
            w_ij=row["usefulness_course"],
            a_j=row["quality_teaching"],
            d_j=row["ease_passing_exam"],
            p_i=row["importance_certainty"],
            lambda_value=row["student_certainty_weight"],
            sigma_j=row["exchange_rate_uncertainty"]
        ),
        axis=1
    )

    total_utility_matrix = df.pivot(
        index="student_id",
        columns="course_id",
        values="total_utility"
    )

    total_utility_matrix.to_csv(output_file)