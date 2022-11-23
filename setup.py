from setuptools import find_packages, setup

setup(
    name='spherical_generative_modeling',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'potpourri3d',
        'trimesh',
        'torch'
    ]
)
