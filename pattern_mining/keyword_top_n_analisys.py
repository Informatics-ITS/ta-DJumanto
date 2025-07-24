import pandas as pd
import os

print("Loading data...")
df = pd.read_csv('keyword_results/individual_keyword_analysis.csv')

print("Calculating averages across eps and threshold for each top_n-keyword combination...")
result = df.groupby(['top_n_candidates', 'attack_type', 'keyword']).agg({
    'top5_rate': 'mean',
    'top10_rate': 'mean', 
    'top20_rate': 'mean'
}).round(4).reset_index()

os.makedirs('average_results', exist_ok=True)
result.to_excel('average_results/keyword_averages_by_topn.xlsx', index=False)

print("\nResults:")
print(result.to_string(index=False))

print(f"\nSaved: average_results/keyword_averages_by_topn.xslx")
print(f"Total combinations: {len(result)}")