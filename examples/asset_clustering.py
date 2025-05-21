"""
Ejemplo de segmentación de activos financieros utilizando clustering.

Este notebook muestra cómo utilizar el módulo de clustering para agrupar
activos financieros con características similares, útil para diversificación
y construcción de portafolios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Añadir el directorio raíz al path para importar el paquete
sys.path.append(os.path.abspath('..'))

# Importar módulos necesarios
from quant_clustering.clustering.market_regime import AssetClusterer
from quant_clustering.data_processing.financial_data import prepare_multi_asset_data
from quant_clustering.evaluation.cluster_metrics import evaluate_clusters
from quant_clustering.visualization.cluster_plots import plot_cluster_profiles

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# 1. Cargar datos financieros de múltiples activos
# ----------------------------------------------
print("Cargando datos de múltiples activos financieros...")

# Lista de símbolos para análisis
# Incluimos acciones de diferentes sectores
symbols = [
    'AAPL',  # Tecnología
    'MSFT',  # Tecnología
    'AMZN',  # Consumo discrecional
    'GOOGL', # Comunicación
    'META',  # Comunicación
    'TSLA',  # Consumo discrecional
    'BRK-B', # Financiero
    'JNJ',   # Salud
    'PG',    # Consumo básico
    'XOM',   # Energía
    'CVX',   # Energía
    'BAC',   # Financiero
    'PFE',   # Salud
    'KO',    # Consumo básico
    'DIS',   # Comunicación
    'MCD',   # Consumo discrecional
    'NKE',   # Consumo discrecional
    'WMT',   # Consumo básico
    'HD',    # Consumo discrecional
    'V'      # Financiero
]

try:
    # Intentar cargar datos de Yahoo Finance
    data_dict = prepare_multi_asset_data(symbols, start_date='2020-01-01', end_date='2023-01-01')
    print(f"Datos cargados para {len(data_dict)} activos")
    
    # Cargar índice de mercado para referencia
    market_index = prepare_multi_asset_data(['SPY'], start_date='2020-01-01', end_date='2023-01-01')
    market_index = market_index.get('SPY')
    
except Exception as e:
    print(f"Error al cargar datos: {e}")
    # Crear datos sintéticos para demostración
    print("Creando datos sintéticos para demostración...")
    
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
    
    # Crear datos sintéticos para cada símbolo
    data_dict = {}
    
    # Definir características por sector
    sectors = {
        'tech': {'drift': 0.0015, 'vol': 0.02, 'symbols': ['AAPL', 'MSFT', 'GOOGL']},
        'consumer': {'drift': 0.0008, 'vol': 0.015, 'symbols': ['AMZN', 'TSLA', 'MCD', 'NKE', 'HD', 'DIS']},
        'financial': {'drift': 0.0005, 'vol': 0.018, 'symbols': ['BRK-B', 'BAC', 'V']},
        'health': {'drift': 0.0006, 'vol': 0.012, 'symbols': ['JNJ', 'PFE']},
        'staples': {'drift': 0.0004, 'vol': 0.01, 'symbols': ['PG', 'KO', 'WMT']},
        'energy': {'drift': 0.0003, 'vol': 0.022, 'symbols': ['XOM', 'CVX']},
        'comm': {'drift': 0.001, 'vol': 0.019, 'symbols': ['META']}
    }
    
    # Generar precios para cada símbolo
    for sector, info in sectors.items():
        base_drift = info['drift']
        base_vol = info['vol']
        
        for symbol in info['symbols']:
            price = 100
            prices = []
            
            # Añadir variación individual
            drift_var = np.random.uniform(-0.0005, 0.0005)
            vol_var = np.random.uniform(-0.005, 0.005)
            
            for _ in range(len(dates)):
                price = price * (1 + np.random.normal(base_drift + drift_var, base_vol + vol_var))
                prices.append(price)
            
            # Crear DataFrame
            data_dict[symbol] = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                'close': prices,
                'volume': [np.random.randint(1000000, 10000000) for _ in range(len(dates))]
            }, index=dates)
    
    # Crear índice de mercado sintético
    price = 100
    prices = []
    for _ in range(len(dates)):
        price = price * (1 + np.random.normal(0.0007, 0.012))
        prices.append(price)
    
    market_index = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(5000000, 50000000) for _ in range(len(dates))]
    }, index=dates)
    
    print(f"Datos sintéticos creados para {len(data_dict)} activos")

# 2. Agrupar activos utilizando clustering
# --------------------------------------
print("\nAgrupando activos financieros...")

# Crear y entrenar el modelo
clusterer = AssetClusterer(n_clusters=5)
clusterer.fit(data_dict, market_index)

# Obtener estadísticas por cluster
cluster_stats = clusterer.get_cluster_stats()
print("\nEstadísticas por cluster:")
print(cluster_stats)

# 3. Visualizar clusters
# -------------------
print("\nVisualizando clusters de activos...")

# Visualizar clusters
fig = clusterer.plot_clusters()
plt.savefig('asset_clusters.png')
plt.close(fig)
print("Gráfico de clusters guardado como 'asset_clusters.png'")

# Visualizar perfiles de clusters
fig = plot_cluster_profiles(clusterer.features_df.drop('cluster', axis=1), clusterer.labels_)
plt.savefig('cluster_profiles.png')
plt.close(fig)
print("Gráfico de perfiles guardado como 'cluster_profiles.png'")

# 4. Análisis de resultados
# -----------------------
print("\nAnálisis de resultados:")

# Identificar el cluster más volátil
if 'avg_volatility' in cluster_stats.columns:
    most_volatile_cluster = cluster_stats['avg_volatility'].idxmax()
    print(f"- Cluster más volátil: {cluster_stats.loc[most_volatile_cluster, 'cluster']}")
    print(f"  Volatilidad media: {cluster_stats.loc[most_volatile_cluster, 'avg_volatility']:.6f}")
    print(f"  Activos: {cluster_stats.loc[most_volatile_cluster, 'symbols']}")

# Identificar el cluster con mejor rendimiento
if 'avg_return' in cluster_stats.columns:
    best_performing_cluster = cluster_stats['avg_return'].idxmax()
    print(f"- Cluster con mejor rendimiento: {cluster_stats.loc[best_performing_cluster, 'cluster']}")
    print(f"  Retorno medio: {cluster_stats.loc[best_performing_cluster, 'avg_return']:.6f}")
    print(f"  Activos: {cluster_stats.loc[best_performing_cluster, 'symbols']}")

# Identificar el cluster con mejor ratio de Sharpe
if 'avg_sharpe' in cluster_stats.columns:
    best_sharpe_cluster = cluster_stats['avg_sharpe'].idxmax()
    print(f"- Cluster con mejor ratio de Sharpe: {cluster_stats.loc[best_sharpe_cluster, 'cluster']}")
    print(f"  Sharpe medio: {cluster_stats.loc[best_sharpe_cluster, 'avg_sharpe']:.6f}")
    print(f"  Activos: {cluster_stats.loc[best_sharpe_cluster, 'symbols']}")

# 5. Aplicaciones para trading cuantitativo
# --------------------------------------
print("\nAplicaciones para trading cuantitativo:")
print("- Diversificación: Seleccionar activos de diferentes clusters para maximizar diversificación")
print("- Rotación sectorial: Identificar clusters con momentum positivo para sobreponderación")
print("- Gestión de riesgo: Limitar exposición a clusters altamente correlacionados")
print("- Construcción de portafolios: Asignar pesos basados en características de clusters")
print("- Pares trading: Identificar activos similares dentro del mismo cluster")

# 6. Conclusiones
# -------------
print("\nConclusiones:")
print("- El clustering de activos financieros permite identificar grupos con características similares")
print("- Esta información es valiosa para la construcción de portafolios diversificados")
print("- Los clusters identificados muestran patrones distintivos en términos de rendimiento y riesgo")
print("- La técnica puede aplicarse a diferentes clases de activos y horizontes temporales")

print("\nEjemplo completado con éxito.")
