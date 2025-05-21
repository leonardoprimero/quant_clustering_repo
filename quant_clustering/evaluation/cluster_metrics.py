"""
Módulos para evaluación y validación de clusters en series temporales financieras.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def evaluate_clusters(features, labels, metrics=None):
    """
    Evalúa la calidad de los clusters utilizando múltiples métricas.
    
    Parameters
    ----------
    features : numpy.ndarray or pandas.DataFrame
        Características utilizadas para clustering.
    labels : numpy.ndarray
        Etiquetas de cluster asignadas.
    metrics : list, optional
        Lista de métricas a calcular. Por defecto, calcula todas las disponibles.
        
    Returns
    -------
    dict
        Diccionario con los valores de las métricas calculadas.
    """
    if metrics is None:
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    
    if isinstance(features, pd.DataFrame):
        features = features.values
    
    results = {}
    
    # Verificar que hay al menos 2 clusters
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return {"error": "Se necesitan al menos 2 clusters para calcular métricas"}
    
    # Calcular métricas
    for metric in metrics:
        try:
            if metric == 'silhouette':
                results['silhouette'] = silhouette_score(features, labels)
            elif metric == 'calinski_harabasz':
                results['calinski_harabasz'] = calinski_harabasz_score(features, labels)
            elif metric == 'davies_bouldin':
                results['davies_bouldin'] = davies_bouldin_score(features, labels)
        except Exception as e:
            results[metric] = f"Error: {str(e)}"
    
    return results


def find_optimal_clusters(features, cluster_range=range(2, 11), method='kmeans', metrics=None):
    """
    Encuentra el número óptimo de clusters utilizando múltiples métricas.
    
    Parameters
    ----------
    features : numpy.ndarray or pandas.DataFrame
        Características para clustering.
    cluster_range : range or list, default=range(2, 11)
        Rango de números de clusters a evaluar.
    method : str, default='kmeans'
        Método de clustering a utilizar.
    metrics : list, optional
        Lista de métricas a calcular.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con los valores de las métricas para cada número de clusters.
    """
    if metrics is None:
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    
    if isinstance(features, pd.DataFrame):
        features = features.values
    
    results = []
    
    for n_clusters in cluster_range:
        try:
            # Aplicar clustering
            if method == 'kmeans':
                from sklearn.cluster import KMeans
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'agglomerative':
                from sklearn.cluster import AgglomerativeClustering
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                raise ValueError(f"Método de clustering '{method}' no soportado")
            
            labels = model.fit_predict(features)
            
            # Evaluar clusters
            eval_results = evaluate_clusters(features, labels, metrics)
            eval_results['n_clusters'] = n_clusters
            results.append(eval_results)
            
        except Exception as e:
            print(f"Error al evaluar {n_clusters} clusters: {str(e)}")
    
    return pd.DataFrame(results)


def plot_evaluation_metrics(evaluation_df, figsize=(12, 8)):
    """
    Visualiza las métricas de evaluación para diferentes números de clusters.
    
    Parameters
    ----------
    evaluation_df : pandas.DataFrame
        DataFrame con los resultados de la evaluación.
    figsize : tuple, default=(12, 8)
        Tamaño de la figura.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figura con la visualización.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(evaluation_df.columns) - 1, 1, figsize=figsize)
        
        if len(evaluation_df.columns) - 1 == 1:
            axes = [axes]
        
        metrics = [col for col in evaluation_df.columns if col != 'n_clusters']
        
        for i, metric in enumerate(metrics):
            axes[i].plot(evaluation_df['n_clusters'], evaluation_df[metric], 'o-', linewidth=2)
            axes[i].set_title(f'Métrica: {metric}')
            axes[i].set_xlabel('Número de clusters')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        print("Matplotlib es necesario para visualizar las métricas.")
        return None


def calculate_dunn_index(features, labels):
    """
    Calcula el índice de Dunn para evaluar la calidad del clustering.
    
    El índice de Dunn es el ratio entre la distancia mínima entre clusters
    y la distancia máxima intra-cluster. Un valor mayor indica mejor clustering.
    
    Parameters
    ----------
    features : numpy.ndarray or pandas.DataFrame
        Características utilizadas para clustering.
    labels : numpy.ndarray
        Etiquetas de cluster asignadas.
        
    Returns
    -------
    float
        Valor del índice de Dunn.
    """
    from scipy.spatial.distance import pdist, squareform
    
    if isinstance(features, pd.DataFrame):
        features = features.values
    
    # Calcular matriz de distancias
    distances = squareform(pdist(features))
    
    # Identificar clusters únicos
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    if n_clusters < 2:
        return 0
    
    # Calcular distancias mínimas entre clusters
    min_between_cluster = float('inf')
    max_within_cluster = 0
    
    for i in range(n_clusters):
        cluster_i_indices = np.where(labels == unique_clusters[i])[0]
        
        # Distancia máxima dentro del cluster i
        if len(cluster_i_indices) > 1:
            within_i = distances[np.ix_(cluster_i_indices, cluster_i_indices)]
            np.fill_diagonal(within_i, 0)
            max_within_i = np.max(within_i)
            max_within_cluster = max(max_within_cluster, max_within_i)
        
        # Distancias mínimas entre clusters
        for j in range(i + 1, n_clusters):
            cluster_j_indices = np.where(labels == unique_clusters[j])[0]
            between_ij = distances[np.ix_(cluster_i_indices, cluster_j_indices)]
            min_between_ij = np.min(between_ij)
            min_between_cluster = min(min_between_cluster, min_between_ij)
    
    # Calcular índice de Dunn
    if max_within_cluster == 0:
        return float('inf')  # Evitar división por cero
    
    return min_between_cluster / max_within_cluster
