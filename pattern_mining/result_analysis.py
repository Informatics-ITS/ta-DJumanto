import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_pattern(pattern_str):
    try:
        if isinstance(pattern_str, str):
            pattern_list = ast.literal_eval(pattern_str)
        else:
            pattern_list = pattern_str
        return frozenset(pattern_list)
    except:
        return frozenset([str(pattern_str)])

def analyze_best_combination(df):
    if 'top_n_candidates' in df.columns:
        df = df.rename(columns={'top_n_candidates': 'top_n'})
    
    df['normalized_pattern'] = df['pattern'].apply(normalize_pattern)
    
    results = []
    
    param_groups = df.groupby(['eps', 'threshold', 'top_n'])
    
    for (eps, threshold, top_n), group in param_groups:
        unique_k5 = group[group['top_k'] == 5]['normalized_pattern'].nunique()
        unique_k10 = group[group['top_k'] == 10]['normalized_pattern'].nunique()
        unique_k20 = group[group['top_k'] == 20]['normalized_pattern'].nunique()
        
        total_unique = group['normalized_pattern'].nunique()
        
        results.append({
            'eps': eps,
            'threshold': threshold, 
            'top_n': top_n,
            'unique_k5': unique_k5,
            'unique_k10': unique_k10,
            'unique_k20': unique_k20,
            'total_unique': total_unique,
            'total_samples': len(group)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_unique', ascending=False)
    
    return results_df

def export_top_patterns(df, results_df, output_path="best_patterns.xlsx"):
    best_params = results_df.iloc[0]
    best_group = df[
        (df['eps'] == best_params['eps']) &
        (df['threshold'] == best_params['threshold']) &
        (df['top_n_candidates'] == best_params['top_n'])
    ]
    
    unique_patterns = best_group.drop_duplicates(subset='normalized_pattern').copy()
    unique_patterns = unique_patterns[['pattern', 'top_k', 'eps', 'threshold', 'top_n_candidates']]
    
    top_20_combinations = results_df
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        unique_patterns.to_excel(writer, sheet_name="Best Unique Patterns", index=False)
        top_20_combinations.to_excel(writer, sheet_name="Top 20 Combinations", index=False)

def visualize_correlation_heatmap(results_df):
    corr_df = results_df[['eps', 'threshold', 'top_n', 'total_unique']].copy()
    
    corr_df['top_n'] = pd.to_numeric(corr_df['top_n'], errors='coerce')
    
    correlation = corr_df.corr(numeric_only=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap: Parameters vs Total Unique")
    plt.tight_layout()
    plt.savefig('pattern_corr.png')

def find_best_parameters(file_path, output_excel="best_patterns.xlsx"):
    df = pd.read_csv(file_path)
    df['normalized_pattern'] = df['pattern'].apply(normalize_pattern)
    
    results = analyze_best_combination(df)
    print(results)

    export_top_patterns(df, results, output_excel)
    visualize_correlation_heatmap(results)
    
    return results

if __name__ == "__main__":
    results = find_best_parameters('keyword_results/pattern_mining_results.csv')
