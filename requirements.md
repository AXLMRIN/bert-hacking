**Alexandre - h200**

- python: 3.13
- cuda: 12.8
- pytorch: 2.11.0+cu128
- transformers: 5.7.0
- datasets:4.8.5
- pandas:3.0.2
- numpy:2.4.4
- sklearn: 1.8.0

```
python -m pip install --upgrade pip setuptools wheel

pip uninstall -y torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Alexandre - icovac**

- python 3.11
- cuda: 12.2
- pytorch: 2.11.0+cu126
- transformers: 5.6.1
- datasets: 4.8.4
- pandas: 3.0.2
- numpy: 2.4.4
- sklearn: 1.8.0

```
python -m pip install --upgrade pip setuptools wheel

pip uninstall -y torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Axel - french-media-database**

- python:3.11
- cuda: 12.7
- pytorch=2.7.1=cuda126_mkl_py311_hcada2b2_300
- transformers==4.52.4
- datasets:3.6.0
- pandas:2.3.0
- numpy:2.3.0
- scikit-learn: 1.7.0
- pyyaml: 6.0.2