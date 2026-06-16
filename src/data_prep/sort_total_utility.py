from pathlib import Path
import pandas as pd


def sort_total_utility(
    input_file: str = "data/total_utility.csv",
    output_file: str = "data/total_utility_sorted.csv"
):
    input_path = Path(input_file)

    df = pd.read_csv(input_path)

    # Первый столбец — это student_id
    student_col = df.columns[0]

    # Превращаем матрицу в длинный формат
    long_df = df.melt(
        id_vars=[student_col],
        var_name="j",
        value_name="u"
    )

    # Переименовываем student_id в i
    long_df = long_df.rename(columns={student_col: "i"})

    # Приводим типы
    long_df["i"] = long_df["i"].astype(int)
    long_df["j"] = long_df["j"].astype(int)
    long_df["u"] = long_df["u"].astype(float)

    # Сортировка по убыванию полезности
    long_df = long_df.sort_values(by="u", ascending=False).reset_index(drop=True)

    # Оставляем только нужный порядок столбцов
    long_df = long_df[["u", "i", "j"]]

    long_df.to_csv(output_file, index=False)

    print(f"Sorted total utility saved to: {Path(output_file).resolve()}")
    print("\nPreview:")
    print(long_df.head(20))


if __name__ == "__main__":
    sort_total_utility()