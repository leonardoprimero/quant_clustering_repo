"""
Módulos para procesamiento y preparación de datos financieros para análisis de clustering.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_financial_data(symbol, start_date=None, end_date=None, source='yahoo'):
    """
    Carga datos financieros desde diversas fuentes.
    
    Parameters
    ----------
    symbol : str
        Símbolo del activo financiero.
    start_date : str, optional
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date : str, optional
        Fecha de fin en formato 'YYYY-MM-DD'.
    source : str, default='yahoo'
        Fuente de datos ('yahoo', 'csv', etc.).
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con datos OHLCV del activo.
    """
    try:
        if source.lower() == 'yahoo':
            import yfinance as yf
            
            # Configurar fechas
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
                
            # Descargar datos
            data = yf.download(symbol, start=start_date, end=end_date)
            
            # Renombrar columnas a minúsculas
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        elif source.lower() == 'csv':
            # Implementar carga desde CSV
            pass
            
        else:
            raise ValueError(f"Fuente de datos '{source}' no soportada.")
            
    except ImportError:
        print("Para usar Yahoo Finance como fuente, instala yfinance: pip install yfinance")
        return None
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None


def prepare_multi_asset_data(symbols, start_date=None, end_date=None, source='yahoo'):
    """
    Prepara datos para múltiples activos financieros.
    
    Parameters
    ----------
    symbols : list
        Lista de símbolos de activos financieros.
    start_date : str, optional
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date : str, optional
        Fecha de fin en formato 'YYYY-MM-DD'.
    source : str, default='yahoo'
        Fuente de datos ('yahoo', 'csv', etc.).
        
    Returns
    -------
    dict
        Diccionario con símbolos como claves y DataFrames como valores.
    """
    data_dict = {}
    
    for symbol in symbols:
        data = load_financial_data(symbol, start_date, end_date, source)
        if data is not None:
            data_dict[symbol] = data
    
    return data_dict


def extract_price_features(price_data, window_sizes=[5, 20, 60]):
    """
    Extrae características técnicas de una serie temporal de precios.
    
    Parameters
    ----------
    price_data : pandas.DataFrame
        DataFrame con datos OHLCV.
    window_sizes : list, default=[5, 20, 60]
        Tamaños de ventana para calcular indicadores.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con características extraídas.
    """
    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame('close')
        
    features = pd.DataFrame(index=price_data.index)
    
    # Retornos
    features['returns'] = price_data['close'].pct_change().fillna(0)
    
    # Volatilidad en diferentes ventanas
    for window in window_sizes:
        features[f'volatility_{window}d'] = features['returns'].rolling(window).std().fillna(0)
        features[f'mean_return_{window}d'] = features['returns'].rolling(window).mean().fillna(0)
    
    # Medias móviles y distancias
    for window in window_sizes:
        sma = price_data['close'].rolling(window).mean()
        features[f'sma_{window}d'] = sma
        features[f'sma_distance_{window}d'] = (price_data['close'] - sma) / sma
    
    # Momentum
    for window in window_sizes:
        features[f'momentum_{window}d'] = price_data['close'].pct_change(window).fillna(0)
    
    # Características adicionales
    if 'volume' in price_data.columns:
        features['volume_change'] = price_data['volume'].pct_change().fillna(0)
        
        # Indicador de acumulación/distribución
        if all(col in price_data.columns for col in ['high', 'low', 'close']):
            clv = ((price_data['close'] - price_data['low']) - (price_data['high'] - price_data['close'])) / (price_data['high'] - price_data['low'])
            clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
            features['ad_line'] = (clv * price_data['volume']).cumsum()
    
    # Indicadores de tendencia
    if 'high' in price_data.columns and 'low' in price_data.columns:
        # ATR (Average True Range)
        tr1 = price_data['high'] - price_data['low']
        tr2 = abs(price_data['high'] - price_data['close'].shift())
        tr3 = abs(price_data['low'] - price_data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14d'] = tr.rolling(14).mean().fillna(0)
    
    return features


def normalize_features(features_df, method='standard'):
    """
    Normaliza características para clustering.
    
    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame con características.
    method : str, default='standard'
        Método de normalización ('standard', 'minmax', 'robust').
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con características normalizadas.
    """
    try:
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Método de normalización '{method}' no soportado.")
        
        # Guardar índice y columnas
        index = features_df.index
        columns = features_df.columns
        
        # Normalizar
        normalized_data = scaler.fit_transform(features_df)
        
        # Devolver como DataFrame
        return pd.DataFrame(normalized_data, index=index, columns=columns)
        
    except ImportError:
        print("scikit-learn es necesario para normalizar características.")
        return features_df
