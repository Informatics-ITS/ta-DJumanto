import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('default')
sns.set_palette("husl")

print("Loading data...")
try:
    df_original = pd.read_csv('keyword_results/individual_keyword_analysis.csv')
    print(f"✓ Original data loaded: {len(df_original)} records")
except FileNotFoundError:
    print("❌ individual_keyword_analysis.csv not found!")
    exit()

# Create output directory
os.makedirs('visualizations', exist_ok=True)

print("Processing data for visualization...")

# Calculate averages across all parameter combinations (eps, threshold, top_n)
print("\n" + "="*60)
print("OVERALL KEYWORD PERFORMANCE ANALYSIS")
print("="*60)

# Calculate overall averages for each keyword across ALL parameter combinations
overall_averages = df_original.groupby(['attack_type', 'keyword']).agg({
    'top5_rate': 'mean',
    'top10_rate': 'mean', 
    'top20_rate': 'mean'
}).round(4).reset_index()

print("Overall keyword performance averages calculated:")
print(overall_averages.to_string(index=False))

# Save overall averages to CSV
overall_averages.to_csv('visualizations/overall_keyword_averages.csv', index=False)
print(f"\n✓ Saved overall averages to: visualizations/overall_keyword_averages.csv")

# Create comprehensive ranking visualization
def create_overall_performance_visualization():
    print("\nCreating overall performance visualization...")
    
    # Sort by top10_rate for ranking
    overall_sorted = overall_averages.sort_values('top10_rate', ascending=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Overall ranking bar chart
    ax1 = plt.subplot(2, 3, 1)
    y_pos = np.arange(len(overall_sorted))
    colors = ['red' if at == 'XSS' else 'blue' for at in overall_sorted['attack_type']]
    
    bars = ax1.barh(y_pos, overall_sorted['top10_rate'], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['keyword'].upper()}" for _, row in overall_sorted.iterrows()])
    ax1.set_xlabel('Average Top-10 Detection Rate')
    ax1.set_title('Keywords Ranked by Overall Top-10 Performance', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # 2. Attack type comparison
    ax2 = plt.subplot(2, 3, 2)
    attack_avg = overall_averages.groupby('attack_type')[['top5_rate', 'top10_rate', 'top20_rate']].mean()
    
    x = np.arange(len(attack_avg.index))
    width = 0.25
    
    ax2.bar(x - width, attack_avg['top5_rate'], width, label='Top-5', alpha=0.8)
    ax2.bar(x, attack_avg['top10_rate'], width, label='Top-10', alpha=0.8)
    ax2.bar(x + width, attack_avg['top20_rate'], width, label='Top-20', alpha=0.8)
    
    ax2.set_xlabel('Attack Type')
    ax2.set_ylabel('Average Detection Rate')
    ax2.set_title('Attack Type Performance Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(attack_avg.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, attack_type in enumerate(attack_avg.index):
        for j, metric in enumerate(['top5_rate', 'top10_rate', 'top20_rate']):
            value = attack_avg.loc[attack_type, metric]
            x_pos = i + (j-1)*width
            ax2.text(x_pos, value + 0.01, f'{value:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Performance distribution
    ax3 = plt.subplot(2, 3, 3)
    overall_averages.boxplot(column=['top5_rate', 'top10_rate', 'top20_rate'], ax=ax3)
    ax3.set_title('Performance Distribution Across All Keywords', fontweight='bold')
    ax3.set_ylabel('Detection Rate')
    ax3.grid(True, alpha=0.3)
    
    # 4. XSS keywords detailed comparison
    ax4 = plt.subplot(2, 3, 4)
    xss_data = overall_averages[overall_averages['attack_type'] == 'XSS'].sort_values('top10_rate')
    if not xss_data.empty:
        x_pos = np.arange(len(xss_data))
        width = 0.25
        
        ax4.bar(x_pos - width, xss_data['top5_rate'], width, label='Top-5', alpha=0.8)
        ax4.bar(x_pos, xss_data['top10_rate'], width, label='Top-10', alpha=0.8)
        ax4.bar(x_pos + width, xss_data['top20_rate'], width, label='Top-20', alpha=0.8)
        
        ax4.set_xlabel('XSS Keywords')
        ax4.set_ylabel('Average Detection Rate')
        ax4.set_title('XSS Keywords Performance Breakdown', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([k.upper() for k in xss_data['keyword']], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. SQLi keywords detailed comparison
    ax5 = plt.subplot(2, 3, 5)
    sqli_data = overall_averages[overall_averages['attack_type'] == 'SQLi'].sort_values('top10_rate')
    if not sqli_data.empty:
        x_pos = np.arange(len(sqli_data))
        width = 0.25
        
        ax5.bar(x_pos - width, sqli_data['top5_rate'], width, label='Top-5', alpha=0.8)
        ax5.bar(x_pos, sqli_data['top10_rate'], width, label='Top-10', alpha=0.8)
        ax5.bar(x_pos + width, sqli_data['top20_rate'], width, label='Top-20', alpha=0.8)
        
        ax5.set_xlabel('SQLi Keywords')
        ax5.set_ylabel('Average Detection Rate')
        ax5.set_title('SQLi Keywords Performance Breakdown', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([k.upper() for k in sqli_data['keyword']], rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Performance heatmap
    ax6 = plt.subplot(2, 3, 6)
    heatmap_data = overall_averages.set_index('keyword')[['top5_rate', 'top10_rate', 'top20_rate']]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6,
                cbar_kws={'label': 'Detection Rate'})
    ax6.set_title('Keywords Performance Heatmap', fontweight='bold')
    ax6.set_ylabel('Keywords')
    
    plt.tight_layout()
    return fig

# Create and save overall performance visualization
fig_overall = create_overall_performance_visualization()
fig_overall.savefig('visualizations/overall_keyword_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.close(fig_overall)

# Create detailed statistics table
def create_detailed_statistics():
    print("\nCreating detailed statistics...")
    
    # Calculate additional statistics
    stats_data = []
    
    for _, row in overall_averages.iterrows():
        keyword_data = df_original[
            (df_original['attack_type'] == row['attack_type']) & 
            (df_original['keyword'] == row['keyword'])
        ]
        
        stats_data.append({
            'attack_type': row['attack_type'],
            'keyword': row['keyword'],
            'avg_top5': row['top5_rate'],
            'avg_top10': row['top10_rate'],
            'avg_top20': row['top20_rate'],
            'std_top5': keyword_data['top5_rate'].std(),
            'std_top10': keyword_data['top10_rate'].std(),
            'std_top20': keyword_data['top20_rate'].std(),
            'min_top10': keyword_data['top10_rate'].min(),
            'max_top10': keyword_data['top10_rate'].max(),
            'range_top10': keyword_data['top10_rate'].max() - keyword_data['top10_rate'].min(),
            'total_experiments': len(keyword_data)
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.round(4)
    
    # Save detailed statistics
    stats_df.to_csv('visualizations/detailed_keyword_statistics.csv', index=False)
    print(f"✓ Saved detailed statistics to: visualizations/detailed_keyword_statistics.csv")
    
    return stats_df

# Create detailed statistics
stats_df = create_detailed_statistics()

# Create performance stability analysis
def create_stability_analysis():
    print("\nCreating stability analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stability ranking (lower std = more stable)
    stability_sorted = stats_df.sort_values('std_top10', ascending=True)
    
    y_pos = np.arange(len(stability_sorted))
    colors = ['red' if at == 'XSS' else 'blue' for at in stability_sorted['attack_type']]
    
    bars1 = ax1.barh(y_pos, stability_sorted['std_top10'], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['keyword'].upper()}" for _, row in stability_sorted.iterrows()])
    ax1.set_xlabel('Standard Deviation of Top-10 Rate')
    ax1.set_title('Keywords Stability Ranking\n(Lower = More Stable)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Performance range analysis
    range_sorted = stats_df.sort_values('range_top10', ascending=False)
    
    y_pos2 = np.arange(len(range_sorted))
    colors2 = ['red' if at == 'XSS' else 'blue' for at in range_sorted['attack_type']]
    
    bars2 = ax2.barh(y_pos2, range_sorted['range_top10'], color=colors2, alpha=0.7)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([f"{row['keyword'].upper()}" for _, row in range_sorted.iterrows()])
    ax2.set_xlabel('Top-10 Rate Range (Max - Min)')
    ax2.set_title('Keywords Performance Range\n(Higher = More Variable)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Add legends
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='XSS'),
                       Patch(facecolor='blue', label='SQLi')]
    ax1.legend(handles=legend_elements, loc='lower right')
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return fig

# Create and save stability analysis
fig_stability = create_stability_analysis()
fig_stability.savefig('visualizations/keyword_stability_analysis.png', dpi=300, bbox_inches='tight')
plt.close(fig_stability)

# Generate comprehensive summary report
def generate_summary_report():
    print("\nGenerating comprehensive summary report...")
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE KEYWORD PERFORMANCE ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Analysis based on {len(df_original)} total experiments")
    report.append(f"Covering {len(df_original['eps'].unique())} epsilon values, {len(df_original['threshold'].unique())} threshold values, {len(df_original['top_n_candidates'].unique())} top_n values")
    report.append("")
    
    # Overall best performers
    report.append("TOP PERFORMING KEYWORDS (Overall Average):")
    report.append("-" * 50)
    top_performers = overall_averages.sort_values('top10_rate', ascending=False).head(5)
    for i, (_, row) in enumerate(top_performers.iterrows(), 1):
        report.append(f"{i}. {row['keyword'].upper()} ({row['attack_type']}): {row['top10_rate']:.4f}")
    report.append("")
    
    # Most stable keywords
    report.append("MOST STABLE KEYWORDS (Lowest Variation):")
    report.append("-" * 50)
    most_stable = stats_df.sort_values('std_top10').head(5)
    for i, (_, row) in enumerate(most_stable.iterrows(), 1):
        report.append(f"{i}. {row['keyword'].upper()} ({row['attack_type']}): std={row['std_top10']:.4f}")
    report.append("")
    
    # Attack type comparison
    report.append("ATTACK TYPE COMPARISON:")
    report.append("-" * 30)
    attack_comparison = overall_averages.groupby('attack_type')[['top5_rate', 'top10_rate', 'top20_rate']].mean()
    for attack_type in attack_comparison.index:
        report.append(f"{attack_type}:")
        report.append(f"  Top-5:  {attack_comparison.loc[attack_type, 'top5_rate']:.4f}")
        report.append(f"  Top-10: {attack_comparison.loc[attack_type, 'top10_rate']:.4f}")
        report.append(f"  Top-20: {attack_comparison.loc[attack_type, 'top20_rate']:.4f}")
        report.append("")
    
    # Performance insights
    report.append("KEY INSIGHTS:")
    report.append("-" * 15)
    
    best_overall = overall_averages.loc[overall_averages['top10_rate'].idxmax()]
    worst_overall = overall_averages.loc[overall_averages['top10_rate'].idxmin()]
    most_stable_kw = stats_df.loc[stats_df['std_top10'].idxmin()]
    most_variable_kw = stats_df.loc[stats_df['range_top10'].idxmax()]
    
    report.append(f"• Best overall keyword: {best_overall['keyword'].upper()} ({best_overall['attack_type']}) - {best_overall['top10_rate']:.4f}")
    report.append(f"• Least performing keyword: {worst_overall['keyword'].upper()} ({worst_overall['attack_type']}) - {worst_overall['top10_rate']:.4f}")
    report.append(f"• Most stable keyword: {most_stable_kw['keyword'].upper()} ({most_stable_kw['attack_type']}) - std={most_stable_kw['std_top10']:.4f}")
    report.append(f"• Most variable keyword: {most_variable_kw['keyword'].upper()} ({most_variable_kw['attack_type']}) - range={most_variable_kw['range_top10']:.4f}")
    
    # Performance distribution
    report.append(f"• Average performance across all keywords: {overall_averages['top10_rate'].mean():.4f}")
    report.append(f"• Performance standard deviation: {overall_averages['top10_rate'].std():.4f}")
    report.append(f"• Performance range: {overall_averages['top10_rate'].min():.4f} - {overall_averages['top10_rate'].max():.4f}")
    
    report.append("")
    report.append("="*80)
    
    # Save report
    with open('visualizations/comprehensive_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("✓ Saved comprehensive report to: visualizations/comprehensive_analysis_report.txt")
    
    # Print summary to console
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Best keyword: {best_overall['keyword'].upper()} ({best_overall['attack_type']}) - {best_overall['top10_rate']:.4f}")
    print(f"Most stable: {most_stable_kw['keyword'].upper()} ({most_stable_kw['attack_type']}) - std={most_stable_kw['std_top10']:.4f}")
    print(f"XSS avg: {attack_comparison.loc['XSS', 'top10_rate']:.4f}")
    print(f"SQLi avg: {attack_comparison.loc['SQLi', 'top10_rate']:.4f}")

# Generate the summary report
generate_summary_report()

print("\n" + "="*60)
print("OVERALL ANALYSIS COMPLETE")
print("="*60)
print("New files created:")
print("✓ visualizations/overall_keyword_averages.csv")
print("✓ visualizations/detailed_keyword_statistics.csv") 
print("✓ visualizations/overall_keyword_performance_analysis.png")
print("✓ visualizations/keyword_stability_analysis.png")
print("✓ visualizations/comprehensive_analysis_report.txt")
print(f"\nTotal keywords analyzed: {len(overall_averages)}")
print(f"Total parameter combinations per keyword: {len(df_original) // len(overall_averages)}")