import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('default')

print("Loading data from Excel sheets...")

try:
    df_balance = pd.read_csv('./visualizations/overall_keyword_averages.csv')
    df_imbalance = pd.read_csv('../pattern_mining_imbalance/visualizations/overall_keyword_averages.csv')
    print(f"Balance data: {len(df_balance)} records")
    print(f"Imbalance data: {len(df_imbalance)} records")
except FileNotFoundError:
    print("Excel file not found!")
    exit()

os.makedirs('keyword_plots', exist_ok=True)

def create_keyword_subplot_grid(df, dataset_name, attack_type):
    filtered_data = df[df['attack_type'] == attack_type].sort_values('top10_rate')
    
    if len(filtered_data) == 0:
        print(f"No {attack_type} keywords found in {dataset_name} dataset")
        return None
    n_keywords = len(filtered_data)
    cols = min(4, n_keywords)  
    rows = (n_keywords + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    fig.suptitle(f'{dataset_name} Dataset - {attack_type} Keywords Performance', 
                 fontsize=16, fontweight='bold')
    
    if n_keywords == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if n_keywords > 1 else [axes]
    else:
        axes = axes.flatten()
    

    edge_color = 'red' if attack_type == 'XSS' else 'blue'
    title_color = 'red' if attack_type == 'XSS' else 'blue'
    

    for i, (_, row) in enumerate(filtered_data.iterrows()):
        ax = axes[i]
        
        rates = [row['top5_rate'], row['top10_rate'], row['top20_rate']]
        labels = ['Top-5', 'Top-10', 'Top-20']
        colors = ['lightblue', 'orange', 'lightgreen']
        
        bars = ax.bar(labels, rates, color=colors, alpha=0.8, 
                     edgecolor=edge_color, linewidth=2)
        
        ax.set_title(f"'{row['keyword']}'", fontweight='bold', color=title_color)
        ax.set_ylabel('Detection Rate')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    

    for i in range(len(filtered_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

print("Creating Balance + XSS plot...")
fig1 = create_keyword_subplot_grid(df_balance, "Balance", "XSS")
if fig1:
    fig1.savefig('keyword_plots/balance_xss_keywords.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

print("Creating Balance + SQLi plot...")
fig2 = create_keyword_subplot_grid(df_balance, "Balance", "SQLi")
if fig2:
    fig2.savefig('keyword_plots/balance_sqli_keywords.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

print("Creating Imbalance + XSS plot...")
fig3 = create_keyword_subplot_grid(df_imbalance, "Imbalance", "XSS")
if fig3:
    fig3.savefig('keyword_plots/imbalance_xss_keywords.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

print("Creating Imbalance + SQLi plot...")
fig4 = create_keyword_subplot_grid(df_imbalance, "Imbalance", "SQLi")
if fig4:
    fig4.savefig('keyword_plots/imbalance_sqli_keywords.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

datasets = [
    (df_balance, "Balance"),
    (df_imbalance, "Imbalance")
]

for df, name in datasets:
    print(f"\n{name.upper()} DATASET:")
    xss_count = len(df[df['attack_type'] == 'XSS'])
    sqli_count = len(df[df['attack_type'] == 'SQLi'])
    print(f"  XSS keywords: {xss_count}")
    print(f"  SQLi keywords: {sqli_count}")
    
    if xss_count > 0:
        best_xss = df[df['attack_type'] == 'XSS'].loc[df[df['attack_type'] == 'XSS']['top10_rate'].idxmax()]
        print(f"  Best XSS: '{best_xss['keyword']}' ({best_xss['top10_rate']:.3f})")
    
    if sqli_count > 0:
        best_sqli = df[df['attack_type'] == 'SQLi'].loc[df[df['attack_type'] == 'SQLi']['top10_rate'].idxmax()]
        print(f"  Best SQLi: '{best_sqli['keyword']}' ({best_sqli['top10_rate']:.3f})")

print(f"\nFiles created:")
print("✓ keyword_plots/balance_xss_keywords.png")
print("✓ keyword_plots/balance_sqli_keywords.png")
print("✓ keyword_plots/imbalance_xss_keywords.png")
print("✓ keyword_plots/imbalance_sqli_keywords.png")