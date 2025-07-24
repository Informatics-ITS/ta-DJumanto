import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
from sklearn.cluster import DBSCAN
from collections import Counter
import re
import json
import string
import ast
import os
import networkx as nx

warnings.filterwarnings('ignore')

def custom_token(text):
    return re.findall(r'\b[a-zA-Z0-9]+\b', text)

def remove_eos(text):
    return text.replace('<eos>', '')

def remove_padding(tok_arr):
    return [tok for tok in tok_arr if tok != -100]

print("Loading data...")
attention_hidden_state = np.load('attention_hidden_state.npy')
attention_weights = np.load('attention_weights.npy')
structure_weights = np.load('structure_weights.npy')
content_vocab = pickle.load(open('both_content_vocab.json', 'rb'))
content_vocab = {v: k for k, v in content_vocab.items()}

preds_df = pd.read_excel('evaluation_ATBiLSTM_2025-06-08_23:58:36.xlsx')
malicious_df = preds_df[preds_df['predicted_label'] == 'Malicious']
true_malicious_df = malicious_df[malicious_df['true_label'] == 'Malicious']
malicious_indices = true_malicious_df.index.tolist()

malicious_hidden_states = attention_hidden_state[malicious_indices]
malicious_weights = attention_weights[malicious_indices]
malicious_structure_weights = structure_weights[malicious_indices]
scaler = StandardScaler()
malicious_hidden_states = scaler.fit_transform(malicious_hidden_states)

eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_values = [0.3, 0.4, 0.5, 0.6, 0.7]
top_n_values = [5, 10, 20]
min_samples = 10

xss_patterns = ['iframe','script','body','img','input', 'alert', 'cookie', '<','>', '(',')']
sqli_patterns = ['select', 'union', 'from', 'limit', 'order', 'where', 'insert','*','(',')','-','\'']

malicious_index = np.array(malicious_indices)
alphanumeric = list(string.ascii_letters + string.digits)
hex_digits = string.hexdigits[:16] + string.hexdigits[16:]
hexes = [f'0x{a}{b}' for a in hex_digits for b in hex_digits] + [f'x{a}{b}' for a in hex_digits for b in hex_digits]


keyword_results = []
clustering_info = []
pattern_results = []

print("Starting parameter exploration...")
print(f"Testing {len(eps_values)} epsilon × {len(threshold_values)} threshold × {len(top_n_values)} top_n combinations")

for eps in eps_values:
    for threshold in threshold_values:
        print(f"\nProcessing: eps={eps}, threshold={threshold}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        cluster_labels = dbscan.fit_predict(malicious_hidden_states)
        
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]
        n_clusters = len(unique_clusters)
        n_noise = np.sum(cluster_labels == -1)
        
        cluster_members = {}
        noise_points = []
        
        for i, label in enumerate(cluster_labels):
            traffic_id = malicious_index[i]
            if label == -1:
                noise_points.append(traffic_id)
            else:
                if label not in cluster_members:
                    cluster_members[label] = []
                cluster_members[label].append(traffic_id)
        
        clustering_info.append({
            'eps': eps,
            'threshold': threshold,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'total_points': len(cluster_labels),
            'cluster_sizes': {f'cluster_{k}': len(v) for k, v in cluster_members.items()},
            'cluster_members': {f'cluster_{k}': v for k, v in cluster_members.items()},
            'noise_points': noise_points,
            'noise_percentage': (n_noise / len(cluster_labels)) * 100
        })
        
        print(f"  Clusters: {n_clusters}, Noise: {n_noise}")
        
        if n_clusters == 0:
            continue
        
        for top_n in top_n_values:
            print(f"    Processing top_n={top_n} candidates...")
            
            keyword_stats = {}
            for pattern in xss_patterns:
                keyword_stats[f'xss_{pattern}'] = {'top5': 0, 'top10': 0, 'top20': 0, 'total': 0}
            for pattern in sqli_patterns:
                keyword_stats[f'sqli_{pattern}'] = {'top5': 0, 'top10': 0, 'top20': 0, 'total': 0}
            
            for cluster_id in unique_clusters:
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_size = len(cluster_indices)
                
                if cluster_size < 5:
                    continue
                
                cluster_weights = [malicious_weights[i] for i in cluster_indices]
                structure_weights_cluster = [malicious_structure_weights[i] for i in cluster_indices]
                
                tokenized_urls = []
                for i in malicious_index[cluster_indices]:
                    tokens = true_malicious_df.loc[i, 'tokenized_url']
                    if isinstance(tokens, str):
                        try:
                            tokens = ast.literal_eval(tokens)
                            tokens = remove_padding(tokens)
                            tokens = [remove_eos(content_vocab[token]) for token in tokens]
                        except:
                            tokens = tokens.split()
                            tokens = [content_vocab[token] for token in tokens]
                    tokenized_urls.append(tokens)

                cluster_documents = [' '.join(tokens) for tokens in tokenized_urls]
                
                stop_words = ['query', 'unk', 'eos', 'page', 'submit', 'php', 'js', 'html', 'png', 'null', 'jsp', 'assets'] + alphanumeric + hexes
                tfidf = TfidfVectorizer(        
                    tokenizer=custom_token,
                    token_pattern=None,
                    use_idf=True,
                    stop_words=stop_words,
                    max_features=100000)
                
                try:
                    tfidf_matrix = tfidf.fit_transform(cluster_documents)
                    feature_names = tfidf.get_feature_names_out()
                    tfidf_sums = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
                    top_n_indices = np.argsort(tfidf_sums)[-min(len(feature_names), top_n):][::-1]
                    candidate_set = [feature_names[i] for i in top_n_indices]
                except:
                    all_tokens = []
                    for tokens in tokenized_urls:
                        all_tokens.extend(tokens)
                    token_counter = Counter(all_tokens)
                    candidate_set = [token for token, _ in token_counter.most_common(top_n)]

                candidate_sets = []
                for i in range(len(cluster_documents)):
                    for token in candidate_set:
                        if token in cluster_documents[i]:
                            candidate_sets.append((malicious_index[cluster_indices[i]], 
                                                 cluster_documents[i], 
                                                 cluster_weights[i], 
                                                 structure_weights_cluster[i]))
                            break
                
                all_top_keywords = []
                exclude_words = {'<eos>', '<query>', '<unk>', 'query', 'unk', '<UNK>', '&', ',', '\\','"', 'assets', '.', 'php', 'js', 'html', 'png'}

                for i in range(len(candidate_sets)):
                    weight = candidate_sets[i][2].flatten()
                    tokens = candidate_sets[i][1].split()
                    structure_weight = candidate_sets[i][3].flatten()

                    min_len = min(len(tokens), len(weight))
                    weights = np.array(weight[:min_len])
                    tokens_array = np.array(tokens[:min_len])
                    sweights = np.array(structure_weight[:min_len])
                    combined_weights = sweights + weights
                    
                    valid_indices = []
                    for idx, token in enumerate(tokens_array):
                        if token not in exclude_words and token.strip():
                            valid_indices.append(idx)

                    if len(valid_indices) == 0:
                        continue
                        
                    valid_tokens = tokens_array[valid_indices]
                    valid_combined_weights = combined_weights[valid_indices]
                    tokens_lower = [token.lower() for token in valid_tokens]
                    
                    for pattern in xss_patterns:
                        if pattern.lower() in tokens_lower:
                            keyword_stats[f'xss_{pattern}']['total'] += 1
                    
                    for pattern in sqli_patterns:
                        if pattern.lower() in tokens_lower:
                            keyword_stats[f'sqli_{pattern}']['total'] += 1
                    
                    for k in [5, 10, 20]:
                        actual_k = min(k, len(valid_tokens))
                        if actual_k > 0:
                            top_k_indices = np.argsort(valid_combined_weights)[-actual_k:][::-1]
                            top_k_tokens = valid_tokens[top_k_indices]
                            top_k_tokens_lower = [token.lower() for token in top_k_tokens]
                            
                            for pattern in xss_patterns:
                                if pattern.lower() in top_k_tokens_lower:
                                    keyword_stats[f'xss_{pattern}'][f'top{k}'] += 1

                            for pattern in sqli_patterns:
                                if pattern.lower() in top_k_tokens_lower:
                                    keyword_stats[f'sqli_{pattern}'][f'top{k}'] += 1

                    top_keywords_by_k = {}
                    for k in [5, 10, 20]:
                        if len(valid_tokens) >= k:
                            top_k_indices = np.argsort(valid_combined_weights)[-k:][::-1]
                            top_k_tokens = valid_tokens[top_k_indices]
                            top_k_weights = valid_combined_weights[top_k_indices]
                            top_keywords_by_k[k] = (candidate_sets[i][0], top_k_tokens, top_k_weights)
                    
                    if top_keywords_by_k:
                        all_top_keywords.append(top_keywords_by_k)
                
                for top_k in [5, 10, 20]:
                    k_keywords = []
                    for keyword_dict in all_top_keywords:
                        if top_k in keyword_dict:
                            traffic_id, tokens, weights = keyword_dict[top_k]
                            k_keywords.append((traffic_id, tokens, weights))
                    
                    if not k_keywords:
                        continue
                        
                    top_m_showed = min(20, top_k)
                    token_counter = Counter(token for _, tokens, _ in k_keywords for token in tokens)
                    top_tokens = [token for token, _ in token_counter.most_common(top_m_showed)]
                    
                    if len(top_tokens) == 0:
                        continue
                        
                    token_to_index = {token: idx for idx, token in enumerate(top_tokens)}
                    top_word_co_matrix = np.zeros((len(top_tokens), len(top_tokens)))

                    for _, tokens, _ in k_keywords:
                        traffic_tokens = set(tokens)
                        for i in top_tokens:
                            for j in top_tokens:
                                if i != j and i in traffic_tokens and j in traffic_tokens:
                                    idx_i = token_to_index[i]
                                    idx_j = token_to_index[j]
                                    top_word_co_matrix[idx_i, idx_j] += 1

                    top_word_co_matrix = (top_word_co_matrix + top_word_co_matrix.T) / 2
                    
                    if np.max(top_word_co_matrix) > 0:
                        top_word_co_matrix = top_word_co_matrix / np.max(top_word_co_matrix)

                    if False:
                        os.makedirs('patterns_images', exist_ok=True)
                        plt.figure(figsize=(15, 12))
                        ax = sns.heatmap(
                            top_word_co_matrix,
                            xticklabels=top_tokens,
                            yticklabels=top_tokens,
                            cmap="Blues",
                            cbar=True,
                            fmt='.2f'
                        )
                        plt.title(f"eps={eps}, th={threshold}, topn={top_n}, cluster={cluster_id}, topk={top_k}")
                        plt.tight_layout()
                        plt.savefig(f'patterns_images/eps{eps}_th{threshold}_topn{top_n}_cluster{cluster_id}_top{top_k}.png')
                        plt.close()
                    
                    G = nx.Graph()
                    for i in range(len(top_tokens)):
                        for j in range(i + 1, len(top_tokens)):
                            if top_word_co_matrix[i, j] >= threshold:
                                G.add_edge(top_tokens[i], top_tokens[j], weight=top_word_co_matrix[i, j])

                    clusters_nx = list(nx.connected_components(G))
                    patterns_discovered = [sorted(list(cluster)) for cluster in clusters_nx if len(cluster) >= 2]
                    
                    if patterns_discovered:
                        for pattern in patterns_discovered:
                            pattern_results.append({
                                'eps': eps,
                                'threshold': threshold,
                                'top_n_candidates': top_n,
                                'cluster_id': cluster_id,
                                'top_k': top_k,
                                'pattern': pattern,
                                'pattern_size': len(pattern),
                                'pattern_str': ', '.join(pattern)
                            })
            
            for pattern in xss_patterns:
                key = f'xss_{pattern}'
                total = keyword_stats[key]['total']
                
                result = {
                    'eps': eps,
                    'threshold': threshold,
                    'top_n_candidates': top_n,
                    'attack_type': 'XSS',
                    'keyword': pattern,
                    'total_occurrences': total,
                    'top5_detected': keyword_stats[key]['top5'],
                    'top10_detected': keyword_stats[key]['top10'],
                    'top20_detected': keyword_stats[key]['top20'],
                    'top5_rate': keyword_stats[key]['top5'] / max(total, 1),
                    'top10_rate': keyword_stats[key]['top10'] / max(total, 1),
                    'top20_rate': keyword_stats[key]['top20'] / max(total, 1)
                }
                keyword_results.append(result)
            
            for pattern in sqli_patterns:
                key = f'sqli_{pattern}'
                total = keyword_stats[key]['total']
                
                result = {
                    'eps': eps,
                    'threshold': threshold,
                    'top_n_candidates': top_n,
                    'attack_type': 'SQLi',
                    'keyword': pattern,
                    'total_occurrences': total,
                    'top5_detected': keyword_stats[key]['top5'],
                    'top10_detected': keyword_stats[key]['top10'],
                    'top20_detected': keyword_stats[key]['top20'],
                    'top5_rate': keyword_stats[key]['top5'] / max(total, 1),
                    'top10_rate': keyword_stats[key]['top10'] / max(total, 1),
                    'top20_rate': keyword_stats[key]['top20'] / max(total, 1)
                }
                keyword_results.append(result)

pd.set_option('display.float_format', '{:.4f}'.format)

print("\nProcessing clustering information...")

clustering_by_eps = {}
for info in clustering_info:
    eps = info['eps']
    if eps not in clustering_by_eps:
        clustering_by_eps[eps] = {
            'n_clusters': [],
            'n_noise': [],
            'total_points': info['total_points'],
            'noise_percentages': [],
            'all_cluster_sizes': []
        }
    
    clustering_by_eps[eps]['n_clusters'].append(info['n_clusters'])
    clustering_by_eps[eps]['n_noise'].append(info['n_noise'])
    clustering_by_eps[eps]['noise_percentages'].append(info['noise_percentage'])
    
    cluster_sizes = [info['cluster_sizes'][key] for key in info['cluster_sizes']]
    clustering_by_eps[eps]['all_cluster_sizes'].extend(cluster_sizes)

clustering_eps_data = []
for eps in sorted(clustering_by_eps.keys()):
    data = clustering_by_eps[eps]
    
    avg_clusters = np.mean(data['n_clusters'])
    avg_noise = np.mean(data['n_noise'])
    avg_noise_pct = np.mean(data['noise_percentages'])
    
    all_sizes = data['all_cluster_sizes']
    if all_sizes:
        min_cluster_size = min(all_sizes)
        max_cluster_size = max(all_sizes)
        avg_cluster_size = np.mean(all_sizes)
        std_cluster_size = np.std(all_sizes)
        median_cluster_size = np.median(all_sizes)
        
        size_counter = Counter(all_sizes)
        common_sizes = [str(size) for size, _ in size_counter.most_common(8)]
        common_sizes_str = ', '.join(common_sizes)
    else:
        min_cluster_size = max_cluster_size = avg_cluster_size = std_cluster_size = median_cluster_size = 0
        common_sizes_str = 'None'
    
    clustering_eps_data.append({
        'eps': eps,
        'avg_n_clusters': round(avg_clusters, 1),
        'avg_n_noise': round(avg_noise, 1),
        'total_points': data['total_points'],
        'avg_noise_percentage': round(avg_noise_pct, 4),
        'min_cluster_size': min_cluster_size,
        'max_cluster_size': max_cluster_size,
        'avg_cluster_size': round(avg_cluster_size, 4),
        'median_cluster_size': round(median_cluster_size, 4),
        'std_cluster_size': round(std_cluster_size, 4),
        'common_cluster_sizes': common_sizes_str
    })

clustering_eps_df = pd.DataFrame(clustering_eps_data)

clustering_simple_data = []
for info in clustering_info:
    cluster_data = {
        'eps': info['eps'],
        'threshold': info['threshold'],
        'n_clusters': info['n_clusters'],
        'n_noise': info['n_noise'],
        'total_points': info['total_points'],
        'noise_percentage': round(info['noise_percentage'], 4)
    }
    
    for i in range(info['n_clusters']):
        cluster_key = f'cluster_{i}'
        if cluster_key in info['cluster_sizes']:
            cluster_data[f'cluster_{i}_size'] = info['cluster_sizes'][cluster_key]
    
    clustering_simple_data.append(cluster_data)

clustering_simple_df = pd.DataFrame(clustering_simple_data)

print("Saving results...")

os.makedirs('keyword_results', exist_ok=True)

keyword_df = pd.DataFrame(keyword_results)
keyword_df.to_csv('keyword_results/individual_keyword_analysis.csv', index=False)

clustering_eps_df.to_csv('keyword_results/clustering_by_epsilon.csv', index=False)
clustering_simple_df.to_csv('keyword_results/clustering_detailed.csv', index=False)

if pattern_results:
    patterns_df = pd.DataFrame(pattern_results)
    patterns_df.to_csv('keyword_results/pattern_mining_results.csv', index=False)
    
    pattern_summary = patterns_df.groupby(['eps', 'threshold', 'top_n_candidates', 'top_k']).agg({
        'pattern': 'count',
        'pattern_size': ['mean', 'min', 'max'],
        'cluster_id': 'nunique'
    }).round(4)
    pattern_summary.columns = ['total_patterns', 'avg_pattern_size', 'min_pattern_size', 'max_pattern_size', 'num_clusters']
    pattern_summary.reset_index().to_csv('keyword_results/pattern_summary_by_params.csv', index=False)

print("- keyword_results/individual_keyword_analysis.csv")
print("- keyword_results/clustering_by_epsilon.csv")
print("- keyword_results/clustering_detailed.csv")
print("- keyword_results/pattern_mining_results.csv")
print("- keyword_results/pattern_summary_by_params.csv")
