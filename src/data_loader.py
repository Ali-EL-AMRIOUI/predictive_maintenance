import os
import pandas as pd


def clean_and_save_data(input_path: str, output_path: str, max_rul: int = 125) -> None:
    """Reads raw C-MAPSS data, assigns column headers, calculates clipped RUL,

    and saves the cleaned DataFrame to the processed directory.
    """
    # 1. Define column names (raw file does not contain headers)
    index_names = ["unit", "cycle"]
    setting_names = ["os1", "os2", "os3"]
    sensor_names = [f"s{i}" for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # 2. Read raw text file
    print(f"Reading raw data from: {input_path}")
    df = pd.read_csv(input_path, sep=r"\s+", header=None, names=col_names)

    # 3. Compute Remaining Useful Life (RUL) and apply piecewise linear cap
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=max_rul)

    # 4. Ensure output directory exists and save processed file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    # Relative file paths from project root
    RAW_PATH = "data/raw/train_FD001.txt"
    PROCESSED_PATH = "data/processed/train_cleaned.csv"

    if os.path.exists(RAW_PATH):
        clean_and_save_data(RAW_PATH, PROCESSED_PATH)
    else:
        print(f"❌ Error: File '{RAW_PATH}' not found. Please download it first!")