from setuptools import setup, find_packages

setup(
    name="ubix",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn==1.5.0",  # 1.5.0
        "scipy==1.13.1",
        "monai==1.3.1",  # instead of 1.0.0
        "matplotlib==3.9.0",
        "seaborn==0.13.2",
        "wandb==0.17.0",
        "numpy==1.26.4",  # "numpy==1.21.2",
        "torch==2.3.0",
        "tqdm==4.66.4",
        "SimpleITK==2.3.1",
        "nibabel==5.2.1",
        "nystrom_attention==0.0.12",
    ],
    entry_points={
        "console_scripts": [],
    },
    author="Coen de Vente",
    author_email="research@coendevente.com",
    description="UBIX",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
