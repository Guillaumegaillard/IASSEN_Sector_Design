# IASSEN Simulator - INSTALLATION GUIDE
## ENVIRONMENT
Tested on a recent linux (Xubuntu) with Git:
 * Linux 5.8.0-59-generic - Xubuntu 20.04.2.0 LTS "Focal Fossa" - Release amd64 (20210209.1)
 * Git version 2.25.1
 
With Anaconda Python distribution and some additional modules (installed using "pip install"):
 * Python 3.8.5 (default, Sep  4 2020, 07:30:14) 

A table from ```conda list```:

 Name             |           Version          | Build          | Channel   | Install
 ---              |           ---              | ---            | ---       | ---
 anaconda         |           2020.11          |        py38_0  |           |
 conda            |           4.9.2            | py38h06a4308_0 |           | 
 func-timeout     |           4.3.5            |         pypi_0 |   pypi    | ```pip install func-timeout```
 matplotlib       |           3.3.2            |              0 |           | 
 numpy            |           1.19.2           | py38h54aff64_0 |           | 
 paramiko         |           2.7.2            |         pypi_0 |   pypi    | ```pip install paramiko```
 pypdf2           |           1.26.0           |         pypi_0 |   pypi    | ```pip install PyPDF2```
 python           |           3.8.5            |     h7579374_1 |           | 
 scp              |           0.13.3           |         pypi_0 |   pypi    | ```pip install scp```
 tabulate         |           0.8.9            |         pypi_0 |   pypi    | ```pip install tabulate```




## GET THE MAIN THING
```bash
git clone https://github.com/Guillaumegaillard/IASSEN_Sector_Design.git
cd IASSEN_Sector_Design
```

## INSTALL DEPENDENCIES

```bash
git clone https://github.com/Guillaumegaillard/Adaptive-Codebook-Optimization.git
cd Adaptive-Codebook-Optimization
git checkout IASSEN 
cd ..
cp Adaptive-Codebook-Optimization/TalonPyCode/ConnectionInfo.mat .
git clone https://github.com/Guillaumegaillard/talon-sector-patterns.git
cd talon-sector-patterns/
git checkout Python_AF
python smooth_the_csv.py 
cd ..
git clone https://github.com/Guillaumegaillard/wrapper_plottings.git
```

## USAGE
Modify the parameters and variables inline and run the scripts.
### Simulation:
```bash
python simu_IASSEN.py
```
### Exploration:
```bash
python sector_explorer.py
```
### Codebook/pattern representation:
```bash
python param_plot.py
```


