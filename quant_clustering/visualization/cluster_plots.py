"""
Herramientas de visualización para análisis de clusters en datos financieros.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates


def plot_market_regimes(price_data, labels, figsize=(14, 10)):
    """
    Visualiza los regímenes de mercado identificados en una serie temporal de precios.
    
    Parameters
    ----------
    price_data : pandas.DataFrame or pandas.Series
        Serie temporal de precios.
    labels : numpy.ndarray
        Etiquetas de cluster asignadas a cada punto temporal.
    figsize : tuple, default=(14, 10)
        Tamaño de la figura.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figura con la visualización.
    """
    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame('close')
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Gráfico de precios
    ax1.plot(price_data.index, price_data['close'], color='black', alpha=0.6)
    ax1.set_title('Regímenes de Mercado Identificados', fontsize=16)
    ax1.set_ylabel('Precio', fontsize=12)
    
    # Colorear fondo según régimen
    cmap = ListedColormap(['lightblue', 'lightgreen', 'salmon', 'lightyellow', 'lightgrey'])
    
    # Calcular estadísticas por régimen
    regime_stats = {}
    for regime in np.unique(labels):
        regime_mask = labels == regime
        regime_returns = price_data['close'].pct_change().fillna(0).loc[regime_mask]
        
        if len(regime_returns) > 0:
            regime_stats[regime] = {
                'count': len(regime_returns),
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': (regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0),
            }
    
    # Crear etiquetas para la leyenda
    legend_labels = []
    for regime in np.unique(labels):
        if regime in regime_stats:
            stats = regime_stats[regime]
            label = f"Régimen {regime}: Ret={stats['mean_return']:.4f}, Vol={stats['volatility']:.4f}"
            legend_labels.append(label)
        else:
            legend_labels.append(f"Régimen {regime}")
    
    # Colorear regímenes en el gráfico de precios
    for regime in np.unique(labels):
        regime_data = price_data.copy()
        regime_data.loc[labels != regime, 'close'] = np.nan
        if not regime_data['close'].isna().all():
            ax1.scatter(regime_data.index, regime_data['close'], 
                       color=cmap.colors[regime % len(cmap.colors)], 
                       s=10, alpha=0.7, label=legend_labels[regime])
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Formatear eje x
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Gráfico de regímenes
    for i in range(len(price_data)):
        if i > 0:
            ax2.axvspan(price_data.index[i-1], price_data.index[i], 
                       color=cmap.colors[labels[i] % len(cmap.colors)], alpha=0.7)
    
    ax2.set_yticks([])
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_title('Secuencia de Regímenes', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_asset_clusters(features_df, labels, figsize=(12, 10), method='pca'):
    """
    Visualiza clusters de activos financieros en un espacio bidimensional.
    
    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame con características de activos.
    labels : numpy.ndarray
        Etiquetas de cluster asignadas.
    figsize : tuple, default=(12, 10)
        Tamaño de la figura.
    method : str, default='pca'
        Método de reducción de dimensionalidad ('pca' o 'tsne').
        
    Returns
    -------
    matplotlib.figure.Figure
        Figura con la visualización.
    """
    # Reducción de dimensionalidad
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Método '{method}' no soportado. Use 'pca' o 'tsne'.")
    
    features_2d = reducer.fit_transform(features_df)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar cada cluster
    for cluster in np.unique(labels):
        cluster_indices = labels == cluster
        ax.scatter(features_2d[cluster_indices, 0], features_2d[cluster_indices, 1], 
                  label=f'Cluster {cluster}', alpha=0.7, s=100)
        
        # Añadir etiquetas para cada punto
        for i, symbol in enumerate(np.array(features_df.index)[cluster_indices]):
            ax.annotate(symbol, (features_2d[cluster_indices, 0][i], 
                                features_2d[cluster_indices, 1][i]),
                       fontsize=9, alpha=0.8)
    
    if method == 'pca':
        ax.set_title('Clustering de Activos Financieros (PCA)', fontsize=16)
        ax.set_xlabel(f'Componente Principal 1', fontsize=12)
        ax.set_ylabel(f'Componente Principal 2', fontsize=12)
    else:
        ax.set_title('Clustering de Activos Financieros (t-SNE)', fontsize=16)
        ax.set_xlabel('Dimensión 1', fontsize=12)
        ax.set_ylabel('Dimensión 2', fontsize=12)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cluster_evaluation(evaluation_df, figsize=(14, 10)):
    """
    Visualiza métricas de evaluación para diferentes números de clusters.
    
    Parameters
    ----------
    evaluation_df : pandas.DataFrame
        DataFrame con resultados de evaluación.
    figsize : tuple, default=(14, 10)
        Tamaño de la figura.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figura con la visualización.
    """
    metrics = [col for col in evaluation_df.columns if col != 'n_clusters']
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        axes[i].plot(evaluation_df['n_clusters'], evaluation_df[metric], 'o-', linewidth=2, markersize=8)
        axes[i].set_title(f'Métrica: {metric}', fontsize=14)
        axes[i].set_xlabel('Número de clusters', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Marcar el mejor valor
        if metric == 'silhouette' or metric == 'calinski_harabasz':
            best_idx = evaluation_df[metric].idxmax()
        else:  # davies_bouldin
            best_idx = evaluation_df[metric].idxmin()
            
        best_n = evaluation_df.loc[best_idx, 'n_clusters']
        best_val = evaluation_df.loc[best_idx, metric]
        
        axes[i].scatter([best_n], [best_val], color='red', s=100, zorder=5)
        axes[i].annotate(f'Mejor: {best_n} clusters',
                        xy=(best_n, best_val),
                        xytext=(10, 20),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    return fig


def plot_cluster_profiles(features_df, labels, figsize=(14, 8)):
    """
    Visualiza perfiles de características para cada cluster.
    
    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame con características.
    labels : numpy.ndarray
        Etiquetas de cluster asignadas.
    figsize : tuple, default=(14, 8)
        Tamaño de la figura.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figura con la visualización.
    """
    # Añadir etiquetas al DataFrame
    df = features_df.copy()
    df['cluster'] = labels
    
    # Calcular medias por cluster
    cluster_means = df.groupby('cluster').mean()
    
    # Normalizar para visualización
    normalized_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar perfiles
    for cluster in normalized_means.index:
        ax.plot(normalized_means.columns, normalized_means.loc[cluster], 
               marker='o', linewidth=2, label=f'Cluster {cluster}')
    
    ax.set_title('Perfiles de Características por Cluster', fontsize=16)
    ax.set_xlabel('Características', fontsize=12)
    ax.set_ylabel('Valor Normalizado', fontsize=12)
    ax.set_xticks(range(len(normalized_means.columns)))
    ax.set_xticklabels(normalized_means.columns, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
