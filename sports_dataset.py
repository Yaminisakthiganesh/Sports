import pandas as pd
import numpy as np

def create_sports_dataset(n=100, filename="sports_dataset.csv"):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(18, 35, n),
        "TrainingHours": np.random.randint(1, 10, n),
        "Height": np.random.randint(160, 200, n),
        "Weight": np.random.randint(55, 100, n),
        "MatchesPlayed": np.random.randint(0, 50, n),
    }

    df = pd.DataFrame(data)
    df["PerformanceScore"] = (
        df["TrainingHours"] * 5 +
        df["MatchesPlayed"] * 2 +
        np.random.normal(0, 10, n)
    ).astype(int)
    df["Selected"] = (df["PerformanceScore"] > 100).astype(int)

    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved to {filename}")

if __name__ == "__main__":
    create_sports_dataset()