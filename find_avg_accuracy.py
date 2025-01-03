import pandas as pd

df = pd.read_csv('ocr_results.csv')

# Ensure 'accuracy_score' is numeric, coercing errors to NaN if necessary
df['license_number_score'] = pd.to_numeric(df['license_number_score'], errors='coerce')

# Removing detected vehicle with no visible number plate
df = df[df['license_number_score'] != 0]

# Get the maximum accuracy score for each car_id
best_accuracies = df.groupby('car_id')['license_number_score'].max()

print(len(best_accuracies))

avg_best_accuracy = best_accuracies.mean()

print(f"Average of best accuracies: {avg_best_accuracy}")
