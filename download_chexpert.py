import pandas as pd

# Load CSVs
train_df = pd.read_csv('/home/cadenug/.cache/kagglehub/datasets/ashery/chexpert/versions/1/train.csv')
valid_df = pd.read_csv('/home/cadenug/.cache/kagglehub/datasets/ashery/chexpert/versions/1/valid.csv')

print(f"Train images: {len(train_df):,}")
print(f"Valid images: {len(valid_df):,}")

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nFirst few rows:")
print(train_df.head())

print("\nLabel columns (diseases):")
disease_cols = [col for col in train_df.columns if col not in ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
print(disease_cols)

# Check label distribution
print("\nDisease counts (train set):")
for col in disease_cols:
    if col in train_df.columns:
        positive = (train_df[col] == 1.0).sum()
        uncertain = (train_df[col] == -1.0).sum()
        negative = (train_df[col] == 0.0).sum()
        missing = train_df[col].isna().sum()
        total = len(train_df)
        print(f"\n{col}:")
        print(f"  Positive: {positive:,} ({positive/total*100:.1f}%)")
        print(f"  Uncertain: {uncertain:,} ({uncertain/total*100:.1f}%)")
        print(f"  Negative: {negative:,} ({negative/total*100:.1f}%)")
        print(f"  Missing: {missing:,} ({missing/total*100:.1f}%)")