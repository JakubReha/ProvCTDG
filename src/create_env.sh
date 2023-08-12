#!/bin/bash
conda create -n ctdg_pyg python=3.9
conda activate ctdg_pyg
conda init bash

conda install gpustat -c conda-forge

conda install --trusted-host pypi.org --trusted-host files.pythonhosted.org natsort
conda install --trusted-host pypi.org --trusted-host files.pythonhosted.org dask
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host https://download.pytorch.org/whl/cu117 torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host data.pyg.org pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host data.pyg.org torch_geometric==2.3.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ray==2.3.0
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pandas==1.5.3
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org tqdm==4.65.0
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org wandb==0.15.0
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org scikit-learn==1.2.2
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org tqdm