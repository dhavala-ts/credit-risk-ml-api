import pandas as pd

# Column names from UCI documentation
columns = [
    "Status_Checking_Account",
    "Duration_Months",
    "Credit_History",
    "Purpose",
    "Credit_Amount",
    "Savings_Account",
    "Employment_Since",
    "Installment_Rate",
    "Personal_Status_Sex",
    "Other_Debtors",
    "Residence_Since",
    "Property",
    "Age",
    "Other_Installment_Plans",
    "Housing",
    "Existing_Credits",
    "Job",
    "People_Liable",
    "Telephone",
    "Foreign_Worker",
    "CreditRisk"
]

# Load space-separated data
df = pd.read_csv(
    "data/german_credit_raw.data",
    sep=" ",
    header=None,
    names=columns
)

# Save clean CSV
df.to_csv("data/german_credit.csv", index=False)

print("Dataset cleaned and saved as data/german_credit.csv")
print(df.head())
