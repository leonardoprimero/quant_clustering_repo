from setuptools import setup, find_packages

setup(
    name="quant_clustering",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "yfinance",
        "scipy"
    ],
    author="Quant Analyst",
    author_email="quant@example.com",
    description="Herramientas de clustering para análisis cuantitativo de trading",
    keywords="clustering, trading, quant, finance, machine learning",
    url="https://github.com/username/quant_clustering_repo",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.7",
)
