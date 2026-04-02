import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("insurance.csv")

print(df.head())

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nAverage Charges by Smoker:")
print(df.groupby("smoker")["charges"].mean())

print("\nAverage Charges by Region:")
print(df.groupby("region")["charges"].mean())

print("\nAverage Charges by Sex:")
print(df.groupby("sex")["charges"].mean())

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

df.groupby('smoker')['charges'].mean().plot(
    kind='bar', ax=axes[0, 0], title='Smoker vs Charges'
)
axes[0, 0].set_ylabel("Charges")

df.groupby('region')['charges'].mean().plot(
    kind='bar', ax=axes[0, 1], title='Region vs Charges'
)
axes[0, 1].set_ylabel("Charges")

axes[1, 0].scatter(df['bmi'], df['charges'])
axes[1, 0].set_title('BMI vs Charges')
axes[1, 0].set_xlabel("BMI")
axes[1, 0].set_ylabel("Charges")

df.boxplot(column='charges', by='smoker', ax=axes[1, 1])
axes[1, 1].set_title('Charges by Smoker')

plt.tight_layout()
plt.show()

print("\nKey Insights:")

smoker_avg = df.groupby("smoker")["charges"].mean()

print(f"Smokers average charges: ${smoker_avg['yes']:.2f}")
print(f"Non-smokers average charges: ${smoker_avg['no']:.2f}")

print("\nSmokers are charged significantly more on average.")