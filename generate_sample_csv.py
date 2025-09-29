import pandas as pd
import numpy as np

def generate_sample_csv(filename="sample_data.csv", n_samples=20):
    np.random.seed(42)

    data = pd.DataFrame({
        "amount": np.random.uniform(1, 20000, n_samples),
        "time": np.random.randint(0, 24, n_samples),
        "loc_encoded": np.random.choice([0, 1, 2, 3], n_samples),  # US=0, EU=1, ASIA=2, AFRICA=3
        "merchant_encoded": np.random.choice([0, 1, 2, 3, 4], n_samples),  # electronics=0 ... others=4
        "device_encoded": np.random.choice([0, 1, 2], n_samples),  # mobile=0, desktop=1, tablet=2
        "previous_transactions": np.random.randint(0, 1000, n_samples)
    })

    data.to_csv(filename, index=False)
    print(f"✅ Sample CSV saved as {filename}")

if __name__ == "__main__":
    generate_sample_csv()
