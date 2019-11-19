<img src="./dtrace/data/drug_crispr_icon.png" width="350" height="350">

![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2530755.svg)](https://10.6084/m9.figshare.10333286)

Integration of drug sensitivity and gene essentiality screens inform on drug mode-of-action

Description
--
This module contains all the Python scripts and results of the [analysis](). ipython notebooks are provided in the 
notebooks directory along with all the generated plots from the analysis. 

Clone code repository
```
git clone https://github.com/EmanuelGoncalves/dtrace.git
```

Get data and unzip from [figshare](https://10.6084/m9.figshare.10336760)
```
cd dtrace

wget https://10.6084/m9.figshare.10336760

gzip data.zip

cd ..
```

Install module and initiate jupyter notebooks
```
python3 setup.py sdist bdist_wheel

pip install -U dist/dtrace-0.5.0-py3-none-any.whl

jupyter notebook
```
