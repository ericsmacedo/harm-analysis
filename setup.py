from setuptools import setup, find_packages

setup(
    name='harm-analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'pytest',
        'matplotlib',
        'scipy',
        'numpy',
        'click',
    ],
    entry_points={
        'console_scripts': [
            'harm_analysis = harm_analysis.cli:cli',
        ],
    },
)
