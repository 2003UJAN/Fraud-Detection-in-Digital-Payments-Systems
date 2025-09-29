import pandas as pd

sample_data = [
    {"amount": 4500, "transaction_type": "Online", "location": "New York", "time": 23, "device_id": 1005},
    {"amount": 200, "transaction_type": "ATM", "location": "London", "time": 14, "device_id": 1002},
    {"amount": 3200, "transaction_type": "Transfer", "location": "Mumbai", "time": 1, "device_id": 1007},
]

df = pd.DataFrame(sample_data)
df.to_csv("sample_transactions.csv", index=False)

print("✅ Sample CSV generated: sample_transactions.csv")

