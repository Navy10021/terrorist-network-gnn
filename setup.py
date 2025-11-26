"""
Terrorist Network Disruption using Temporal GNN
================================================

Setup script for package installation.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Terrorist Network GNN - Advanced Temporal Graph Neural Networks"

# Read requirements
def read_requirements():
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        # Fallback to minimum requirements
        return [
            'torch>=2.0.0',
            'torch-geometric>=2.3.0',
            'numpy>=1.24.0',
            'scipy>=1.10.0',
            'pandas>=2.0.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'networkx>=3.1',
            'tqdm>=4.65.0',
            'scikit-learn>=1.2.0',
        ]

setup(
    name='terrorist-network-gnn',
    version='1.0.0',
    author='Yoon-Seop Lee',
    author_email='iyunseob4@gmail.com',
    description='Advanced Temporal Graph Neural Networks for Terrorist Network Disruption',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/Navy10021/terrorist-network-gnn',
    project_urls={
        'Bug Tracker': 'https://github.com/Navy10021/terrorist-network-gnn/issues',
        'Documentation': 'https://Navy10021.github.io/terrorist-network-gnn/',
        'Source Code': 'https://github.com/Navy10021/terrorist-network-gnn',
    },
    packages=['src'],
    package_dir={'src': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.0.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.2.0',
            'sphinx>=6.2.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        'notebook': [
            'jupyter>=1.0.0',
            'ipykernel>=6.22.0',
            'ipywidgets>=8.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'tgnn-train=scripts.run_experiment:main',
            'tgnn-evaluate=scripts.evaluate_model:main',
            'tgnn-visualize=scripts.visualize_results:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'graph neural networks',
        'temporal networks',
        'network security',
        'critical node detection',
        'deep learning',
        'pytorch',
    ],
)
