"""
Implementación de algoritmos de clustering adaptados para series temporales financieras.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


class MarketRegimeClusterer:
    """
    Clase para identificar regímenes de mercado mediante clustering de series temporales financieras.
    
    Esta clase implementa un enfoque de clustering para identificar diferentes regímenes
    o estados del mercado basados en características extraídas de series temporales financieras.
    
    Parameters
    ----------
    n_clusters : int, default=4
        Número de regímenes o clusters a identificar.
    
    window_size : int, default=20
        Tamaño de la ventana para calcular características.
    
    random_state : int, default=42
        Semilla para reproducibilidad.
    
    Attributes
    ----------
    model : KMeans
        Modelo de clustering entrenado.
    
    scaler : StandardScaler
        Escalador para normalizar características.
    """
    
    def __init__(self, n_clusters=4, window_size=20, random_state=42):
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        
    def _extract_features(self, price_data):
        """
        Extrae características relevantes de la serie temporal de precios.
        
        Parameters
        ----------
        price_data : pandas.DataFrame or pandas.Series
            Serie temporal de precios.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame con las características extraídas.
        """
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame('close')
            
        # Calcular retornos
        returns = price_data['close'].pct_change().fillna(0)
        
        # Características
        features = pd.DataFrame(index=price_data.index)
        
        # Volatilidad (desviación estándar de retornos)
        features['volatility'] = returns.rolling(self.window_size).std().fillna(0)
        
        # Tendencia (retorno acumulado)
        features['trend'] = returns.rolling(self.window_size).mean().fillna(0)
        
        # Momentum (retorno de n días)
        features['momentum'] = returns.rolling(self.window_size).sum().fillna(0)
        
        # Ratio de Sharpe simplificado
        features['sharpe'] = (features['trend'] / features['volatility']).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Distancia a la media móvil
        sma = price_data['close'].rolling(self.window_size).mean().fillna(method='bfill')
        features['sma_distance'] = (price_data['close'] - sma) / sma
        
        return features.fillna(0)
    
    def fit(self, price_data):
        """
        Entrena el modelo de clustering con datos de precios.
        
        Parameters
        ----------
        price_data : pandas.DataFrame or pandas.Series
            Serie temporal de precios.
            
        Returns
        -------
        self : object
            Instancia ajustada.
        """
        features = self._extract_features(price_data)
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)
        return self
    
    def predict(self, price_data):
        """
        Predice los regímenes de mercado para los datos proporcionados.
        
        Parameters
        ----------
        price_data : pandas.DataFrame or pandas.Series
            Serie temporal de precios.
            
        Returns
        -------
        numpy.ndarray
            Etiquetas de cluster asignadas a cada punto temporal.
        """
        features = self._extract_features(price_data)
        scaled_features = self.scaler.transform(features)
        return self.model.predict(scaled_features)
    
    def fit_predict(self, price_data):
        """
        Entrena el modelo y predice los regímenes de mercado.
        
        Parameters
        ----------
        price_data : pandas.DataFrame or pandas.Series
            Serie temporal de precios.
            
        Returns
        -------
        numpy.ndarray
            Etiquetas de cluster asignadas a cada punto temporal.
        """
        self.fit(price_data)
        return self.predict(price_data)
    
    def get_regime_stats(self, price_data, labels):
        """
        Calcula estadísticas para cada régimen identificado.
        
        Parameters
        ----------
        price_data : pandas.DataFrame or pandas.Series
            Serie temporal de precios.
        labels : numpy.ndarray
            Etiquetas de cluster asignadas a cada punto temporal.
            
        Returns
        -------
        pandas.DataFrame
            Estadísticas por régimen.
        """
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame('close')
            
        returns = price_data['close'].pct_change().fillna(0)
        
        # Crear DataFrame con retornos y etiquetas
        regime_data = pd.DataFrame({
            'returns': returns,
            'regime': labels
        })
        
        # Calcular estadísticas por régimen
        stats = []
        for regime in range(self.n_clusters):
            regime_returns = regime_data[regime_data['regime'] == regime]['returns']
            
            if len(regime_returns) > 0:
                stats.append({
                    'regime': regime,
                    'count': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe': (regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0),
                    'max_return': regime_returns.max(),
                    'min_return': regime_returns.min(),
                })
        
        return pd.DataFrame(stats)
    
    def plot_regimes(self, price_data, labels, figsize=(12, 8)):
        """
        Visualiza los regímenes identificados en la serie temporal de precios.
        
        Parameters
        ----------
        price_data : pandas.DataFrame or pandas.Series
            Serie temporal de precios.
        labels : numpy.ndarray
            Etiquetas de cluster asignadas a cada punto temporal.
        figsize : tuple, default=(12, 8)
            Tamaño de la figura.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figura con la visualización.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            
            if isinstance(price_data, pd.Series):
                price_data = price_data.to_frame('close')
            
            # Crear figura
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
            
            # Gráfico de precios
            ax1.plot(price_data.index, price_data['close'], color='black', alpha=0.6)
            ax1.set_title('Regímenes de Mercado Identificados')
            ax1.set_ylabel('Precio')
            
            # Colorear fondo según régimen
            cmap = ListedColormap(['lightblue', 'lightgreen', 'salmon', 'lightyellow', 'lightgrey'])
            
            # Obtener estadísticas por régimen
            stats = self.get_regime_stats(price_data, labels)
            
            # Crear etiquetas para la leyenda
            legend_labels = []
            for regime in range(self.n_clusters):
                if regime in stats['regime'].values:
                    regime_stat = stats[stats['regime'] == regime].iloc[0]
                    label = f"Régimen {regime}: Ret={regime_stat['mean_return']:.4f}, Vol={regime_stat['volatility']:.4f}"
                    legend_labels.append(label)
                else:
                    legend_labels.append(f"Régimen {regime}")
            
            # Colorear regímenes en el gráfico de precios
            for regime in range(self.n_clusters):
                regime_data = price_data.copy()
                regime_data.loc[labels != regime, 'close'] = np.nan
                if not regime_data['close'].isna().all():
                    ax1.scatter(regime_data.index, regime_data['close'], 
                               color=cmap.colors[regime % len(cmap.colors)], 
                               s=10, alpha=0.7, label=legend_labels[regime])
            
            ax1.legend(loc='upper left')
            
            # Gráfico de regímenes
            for i in range(len(price_data)):
                if i > 0:
                    ax2.axvspan(price_data.index[i-1], price_data.index[i], 
                               color=cmap.colors[labels[i] % len(cmap.colors)], alpha=0.7)
            
            ax2.set_yticks([])
            ax2.set_xlabel('Fecha')
            ax2.set_title('Secuencia de Regímenes')
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            print("Matplotlib es necesario para visualizar los regímenes.")
            return None


class AssetClusterer:
    """
    Clase para agrupar activos financieros basados en características similares.
    
    Esta clase implementa un enfoque de clustering para identificar grupos de activos
    con comportamientos similares, útil para diversificación y construcción de portafolios.
    
    Parameters
    ----------
    n_clusters : int, default=5
        Número de grupos de activos a identificar.
    
    feature_window : int, default=252
        Ventana temporal para calcular características (por defecto un año de trading).
    
    random_state : int, default=42
        Semilla para reproducibilidad.
    
    Attributes
    ----------
    model : KMeans
        Modelo de clustering entrenado.
    
    scaler : StandardScaler
        Escalador para normalizar características.
    """
    
    def __init__(self, n_clusters=5, feature_window=252, random_state=42):
        self.n_clusters = n_clusters
        self.feature_window = feature_window
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        
    def _extract_asset_features(self, price_data_dict):
        """
        Extrae características relevantes de múltiples series temporales de activos.
        
        Parameters
        ----------
        price_data_dict : dict
            Diccionario con símbolos de activos como claves y DataFrames/Series como valores.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame con las características extraídas por activo.
        """
        features_list = []
        
        for symbol, data in price_data_dict.items():
            if isinstance(data, pd.Series):
                data = data.to_frame('close')
                
            # Calcular retornos
            returns = data['close'].pct_change().fillna(0)
            
            # Extraer características
            features = {
                'symbol': symbol,
                'volatility': returns[-self.feature_window:].std(),
                'mean_return': returns[-self.feature_window:].mean(),
                'sharpe': returns[-self.feature_window:].mean() / returns[-self.feature_window:].std() if returns[-self.feature_window:].std() > 0 else 0,
                'max_drawdown': (data['close'] / data['close'].cummax() - 1).min(),
                'beta': None,  # Se calculará si se proporciona un índice de referencia
                'correlation': None  # Se calculará si se proporciona un índice de referencia
            }
            
            # Características adicionales
            features['skewness'] = returns[-self.feature_window:].skew() if hasattr(returns, 'skew') else 0
            features['kurtosis'] = returns[-self.feature_window:].kurtosis() if hasattr(returns, 'kurtosis') else 0
            
            features_list.append(features)
            
        return pd.DataFrame(features_list).set_index('symbol')
    
    def fit(self, price_data_dict, market_index=None):
        """
        Entrena el modelo de clustering con datos de precios de múltiples activos.
        
        Parameters
        ----------
        price_data_dict : dict
            Diccionario con símbolos de activos como claves y DataFrames/Series como valores.
        market_index : pandas.DataFrame or pandas.Series, optional
            Serie temporal del índice de mercado para calcular beta y correlaciones.
            
        Returns
        -------
        self : object
            Instancia ajustada.
        """
        # Extraer características
        features_df = self._extract_asset_features(price_data_dict)
        
        # Calcular beta y correlación si se proporciona un índice de mercado
        if market_index is not None:
            if isinstance(market_index, pd.Series):
                market_index = market_index.to_frame('close')
                
            market_returns = market_index['close'].pct_change().fillna(0)
            
            for symbol, data in price_data_dict.items():
                if isinstance(data, pd.Series):
                    data = data.to_frame('close')
                    
                asset_returns = data['close'].pct_change().fillna(0)
                
                # Asegurar que los índices coincidan
                common_idx = asset_returns.index.intersection(market_returns.index)
                if len(common_idx) > 0:
                    asset_returns = asset_returns.loc[common_idx]
                    market_returns_aligned = market_returns.loc[common_idx]
                    
                    # Calcular beta usando regresión lineal
                    try:
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        model.fit(market_returns_aligned.values.reshape(-1, 1), asset_returns.values.reshape(-1, 1))
                        features_df.loc[symbol, 'beta'] = model.coef_[0][0]
                    except:
                        features_df.loc[symbol, 'beta'] = np.nan
                    
                    # Calcular correlación
                    features_df.loc[symbol, 'correlation'] = asset_returns.corr(market_returns_aligned)
        
        # Eliminar columnas con NaN y rellenar los NaN restantes
        features_df = features_df.fillna(0)
        
        # Escalar características
        self.feature_names = features_df.columns.tolist()
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Entrenar modelo
        self.model.fit(scaled_features)
        
        # Guardar resultados
        self.labels_ = self.model.labels_
        self.features_df = features_df
        self.features_df['cluster'] = self.labels_
        
        return self
    
    def predict(self, price_data_dict, market_index=None):
        """
        Predice los clusters para nuevos activos.
        
        Parameters
        ----------
        price_data_dict : dict
            Diccionario con símbolos de activos como claves y DataFrames/Series como valores.
        market_index : pandas.DataFrame or pandas.Series, optional
            Serie temporal del índice de mercado para calcular beta y correlaciones.
            
        Returns
        -------
        pandas.Series
            Serie con los clusters asignados a cada activo.
        """
        features_df = self._extract_asset_features(price_data_dict)
        
        # Calcular beta y correlación si se proporciona un índice de mercado
        if market_index is not None:
            # Código similar al de fit para calcular beta y correlación
            pass
        
        # Asegurar que tenemos las mismas columnas que en el entrenamiento
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.feature_names].fillna(0)
        
        # Escalar y predecir
        scaled_features = self.scaler.transform(features_df)
        labels = self.model.predict(scaled_features)
        
        return pd.Series(labels, index=features_df.index, name='cluster')
    
    def get_cluster_stats(self):
        """
        Obtiene estadísticas para cada cluster identificado.
        
        Returns
        -------
        pandas.DataFrame
            Estadísticas por cluster.
        """
        if not hasattr(self, 'features_df'):
            raise ValueError("El modelo debe ser entrenado primero usando fit().")
        
        stats = []
        for cluster in range(self.n_clusters):
            cluster_data = self.features_df[self.features_df['cluster'] == cluster]
            
            if len(cluster_data) > 0:
                stats.append({
                    'cluster': cluster,
                    'count': len(cluster_data),
                    'symbols': ', '.join(cluster_data.index.tolist()),
                    'avg_volatility': cluster_data['volatility'].mean(),
                    'avg_return': cluster_data['mean_return'].mean(),
                    'avg_sharpe': cluster_data['sharpe'].mean(),
                })
                
                # Añadir beta y correlación si están disponibles
                if 'beta' in cluster_data.columns:
                    stats[-1]['avg_beta'] = cluster_data['beta'].mean()
                if 'correlation' in cluster_data.columns:
                    stats[-1]['avg_correlation'] = cluster_data['correlation'].mean()
        
        return pd.DataFrame(stats)
    
    def plot_clusters(self, figsize=(12, 10)):
        """
        Visualiza los clusters de activos en un espacio bidimensional.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 10)
            Tamaño de la figura.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figura con la visualización.
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            if not hasattr(self, 'features_df'):
                raise ValueError("El modelo debe ser entrenado primero usando fit().")
            
            # Aplicar PCA para visualización en 2D
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(self.scaler.transform(self.features_df.drop('cluster', axis=1)))
            
            # Crear figura
            fig, ax = plt.subplots(figsize=figsize)
            
            # Graficar cada cluster
            for cluster in range(self.n_clusters):
                cluster_indices = self.features_df['cluster'] == cluster
                ax.scatter(features_2d[cluster_indices, 0], features_2d[cluster_indices, 1], 
                          label=f'Cluster {cluster}', alpha=0.7, s=100)
                
                # Añadir etiquetas para cada punto
                for i, symbol in enumerate(self.features_df.index[cluster_indices]):
                    ax.annotate(symbol, (features_2d[cluster_indices, 0].iloc[i], 
                                        features_2d[cluster_indices, 1].iloc[i]),
                               fontsize=9, alpha=0.8)
            
            ax.set_title('Clustering de Activos Financieros')
            ax.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
            ax.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            print("Matplotlib y scikit-learn son necesarios para visualizar los clusters.")
            return None
