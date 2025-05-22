# Quant Clustering Toolkit 🧠🔥

> “If you want to survive in the market jungle, you’d better learn to cluster before the bears find you.”

## What is this?

Welcome to **Quant Clustering Toolkit**: a Python playground where clustering is more than an algorithm—it's your new street sense for financial markets.  
I built this repo because market regimes come and go faster than crypto memes, and I got tired of pretending that all price charts are the same animal.

Here, you'll find tools to spot hidden patterns, detect market shifts, and group assets like you're organizing a barbecue (except nobody brings bad wine).

## Why bother?

- **Find market regimes**: Are we in a bull run or a bear hug? Let the clusters speak.
- **Group assets by vibe**: Maybe tech stocks are partying while oil is sulking. You’ll see it here.
- **Spot anomalies**: Because sometimes a candle just wants attention.
- **Portfolio diversification**: Like mixing Fernet with Coke—diversification is about balance.
- **Better trading decisions**: Algorithms can’t cry, but you’ll sleep better.

## Features

- Clustering algorithms tuned for time series and finance (KMeans, Spectral, DBSCAN, and more)
- Tools to validate clusters—no black boxes here
- Pretty visualizations that even a recruiter will love
- Plug-and-play notebooks with real market data
- Docs and examples with comments that make sense (mostly)

## Repo Structure

```
quant_clustering_repo/
├── data/              # Sample datasets
├── docs/              # Documentation
├── examples/          # Playable notebooks
├── quant_clustering/  # The real code
│   ├── clustering/
│   ├── data_processing/
│   ├── evaluation/
│   ├── features/
│   ├── utils/
│   └── visualization/
├── tests/             # Unit tests (yeah, it has tests)
├── requirements.txt   # Python dependencies
└── setup.py           # Install config
```

## Install

```bash
git clone https://github.com/leonardoprimero/quant_clustering_repo.git
cd quant_clustering_repo
pip install -e .
```

## Quick Start

```python
from quant_clustering.clustering import MarketRegimeClusterer
from quant_clustering.data_processing import load_financial_data

# Load SPY data (because who doesn’t love S&P500?)
data = load_financial_data('SPY', start_date='2020-01-01', end_date='2022-12-31')

# Cluster the regimes (choose your own adventure)
clusterer = MarketRegimeClusterer(n_clusters=4)
labels = clusterer.fit_predict(data)

# Plot like you mean it
clusterer.plot_regimes(data, labels)
```

## License

MIT. Use it, fork it, break it, fix it—just don’t blame me if you lose money.

## Author

Leonardo I (a.k.a. @leonardoprimero)

---

**Pro-tip:**  
If you want a more philosophical explanation about why clustering matters, DM me or check my blog. Otherwise, run the code and see what the market is really telling you.

