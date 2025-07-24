import pandas as pd
import os

print("Loading data...")
df = pd.read_csv('keyword_results/individual_keyword_analysis.csv')

print("Calculating averages across top_n and threshold for each eps-keyword combination...")
result = df.groupby(['eps', 'attack_type', 'keyword']).agg({
    'top5_rate': 'mean',
    'top10_rate': 'mean', 
    'top20_rate': 'mean'
}).round(4).reset_index()

os.makedirs('average_results', exist_ok=True)
result.to_csv('average_results/keyword_averages_by_eps.csv', index=False)

print("\nResults:")
print(result.to_string(index=False))

print(f"\nSaved: average_results/keyword_averages_by_eps.csv")
print(f"Total combinations: {len(result)}")