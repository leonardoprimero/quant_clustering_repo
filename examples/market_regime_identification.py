"""
Ejemplo de identificación de regímenes de mercado utilizando clustering.

Este notebook muestra cómo utilizar el módulo de clustering para identificar
diferentes regímenes o estados del mercado en una serie temporal financiera.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Añadir el directorio raíz al path para importar el paquete
sys.path.append(os.path.abspath('..'))

# Importar módulos necesarios
from quant_clustering.clustering.market_regime import MarketRegimeClusterer
from quant_clustering.data_processing.financial_data import load_financial_data
from quant_clustering.evaluation.cluster_metrics import find_optimal_clusters
from quant_clustering.visualization.cluster_plots import plot_cluster_evaluation

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# 1. Cargar datos financieros
# --------------------------
# Cargar datos del S&P 500 para los últimos 5 años
print("Cargando datos financieros...")
try:
    # Intentar cargar datos de Yahoo Finance
    data = load_financial_data('SPY', start_date='2018-01-01', end_date='2023-01-01')
    print(f"Datos cargados: {len(data)} registros")
    
    # Mostrar primeras filas
    print("\nPrimeras filas de los datos:")
    print(data.head())
    
except Exception as e:
    print(f"Error al cargar datos: {e}")
    # Crear datos sintéticos para demostración
    print("Creando datos sintéticos para demostración...")
    
    np.random.seed(42)
    dates = pd.date_range(start='2018-01-01', end='2023-01-01', freq='B')
    
    # Simular diferentes regímenes
    n_days = len(dates)
    price = 100
    prices = [price]
    
    # Parámetros para diferentes regímenes
    regimes = [
        {'drift': 0.001, 'vol': 0.005, 'days': int(n_days * 0.2)},  # Tendencia alcista suave
        {'drift': -0.002, 'vol': 0.015, 'days': int(n_days * 0.15)},  # Corrección
        {'drift': 0.0005, 'vol': 0.008, 'days': int(n_days * 0.3)},  # Consolidación
        {'drift': -0.003, 'vol': 0.025, 'days': int(n_days * 0.1)},  # Crisis
        {'drift': 0.002, 'vol': 0.01, 'days': int(n_days * 0.25)}    # Recuperación
    ]
    
    for regime in regimes:
        for _ in range(regime['days']):
            price = price * (1 + np.random.normal(regime['drift'], regime['vol']))
            prices.append(price)
    
    # Ajustar longitud si es necesario
    prices = prices[:n_days]
    
    # Crear DataFrame
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 10000000) for _ in range(n_days)]
    }, index=dates)
    
    print(f"Datos sintéticos creados: {len(data)} registros")
    print("\nPrimeras filas de los datos sintéticos:")
    print(data.head())

# 2. Encontrar número óptimo de clusters
# -------------------------------------
print("\nBuscando número óptimo de clusters...")

# Extraer características para clustering
from quant_clustering.data_processing.financial_data import extract_price_features
features = extract_price_features(data)

# Normalizar características
from quant_clustering.data_processing.financial_data import normalize_features
normalized_features = normalize_features(features)

# Encontrar número óptimo de clusters
evaluation_results = find_optimal_clusters(
    normalized_features, 
    cluster_range=range(2, 11),
    method='kmeans'
)

print("\nResultados de evaluación de clusters:")
print(evaluation_results)

# Visualizar resultados de evaluación
fig = plot_cluster_evaluation(evaluation_results)
plt.savefig('cluster_evaluation.png')
plt.close(fig)
print("Gráfico de evaluación guardado como 'cluster_evaluation.png'")

# Determinar número óptimo de clusters
# Podemos usar el índice de silueta como criterio
if 'silhouette' in evaluation_results.columns:
    optimal_clusters = evaluation_results.loc[evaluation_results['silhouette'].idxmax(), 'n_clusters']
else:
    # Valor por defecto si no se puede determinar
    optimal_clusters = 4

print(f"\nNúmero óptimo de clusters: {optimal_clusters}")

# 3. Identificar regímenes de mercado
# ---------------------------------
print("\nIdentificando regímenes de mercado...")

# Crear y entrenar el modelo
clusterer = MarketRegimeClusterer(n_clusters=optimal_clusters)
labels = clusterer.fit_predict(data)

# Obtener estadísticas por régimen
regime_stats = clusterer.get_regime_stats(data, labels)
print("\nEstadísticas por régimen:")
print(regime_stats)

# 4. Visualizar regímenes
# ---------------------
print("\nVisualizando regímenes de mercado...")

# Visualizar regímenes
fig = clusterer.plot_regimes(data, labels)
plt.savefig('market_regimes.png')
plt.close(fig)
print("Gráfico de regímenes guardado como 'market_regimes.png'")

# 5. Análisis de resultados
# -----------------------
print("\nAnálisis de resultados:")

# Identificar el régimen más volátil
most_volatile_regime = regime_stats['volatility'].idxmax()
print(f"- Régimen más volátil: {most_volatile_regime}")
print(f"  Volatilidad: {regime_stats.loc[most_volatile_regime, 'volatility']:.6f}")
print(f"  Retorno medio: {regime_stats.loc[most_volatile_regime, 'mean_return']:.6f}")

# Identificar el régimen con mejor rendimiento
best_performing_regime = regime_stats['mean_return'].idxmax()
print(f"- Régimen con mejor rendimiento: {best_performing_regime}")
print(f"  Retorno medio: {regime_stats.loc[best_performing_regime, 'mean_return']:.6f}")
print(f"  Volatilidad: {regime_stats.loc[best_performing_regime, 'volatility']:.6f}")

# Identificar el régimen con peor rendimiento
worst_performing_regime = regime_stats['mean_return'].idxmin()
print(f"- Régimen con peor rendimiento: {worst_performing_regime}")
print(f"  Retorno medio: {regime_stats.loc[worst_performing_regime, 'mean_return']:.6f}")
print(f"  Volatilidad: {regime_stats.loc[worst_performing_regime, 'volatility']:.6f}")

# 6. Conclusiones
# -------------
print("\nConclusiones:")
print("- La identificación de regímenes de mercado permite adaptar estrategias de trading")
print("  a diferentes condiciones de mercado.")
print("- Los regímenes identificados muestran características distintivas en términos")
print("  de rendimiento y volatilidad.")
print("- Esta información puede utilizarse para ajustar la exposición al riesgo o")
print("  seleccionar diferentes estrategias según el régimen actual.")

print("\nEjemplo completado con éxito.")
