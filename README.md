# SAE-LSTM-based ROM for turbulent flows

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


Nonlinear model order reduction of engineering turbulence using data-assisted neural networks

## Table of Contents

- [Background](#background)
- [Data](#data)
- [Install](#install)
- [Modelling](#Model-training)
- [Methodology](#methodology)
- [License](#license)


## Background

Conducting repeated high-fidelity simulations of complex turbulent flows entails substantial computational costs in engineering applications. Reduced-order modeling (ROM) seeks to derive low-dimensional representations from full-order numerical systems, thereby facilitating rapid forecasting of future flow states. This study presents a novel data-assisted framework that employs deep neural networks for nonlinear ROM of engineering turbulent flows. Specifically, the Stacked Auto-Encoder (SAE) network is utilized for nonlinear dimensionality reduction and feature extraction; the resulting latent features subsequently serve as inputs to the Long Short-Term Memory (LSTM) network for predictive ROM of turbulent fluid dynamics. A comparative analysis is conducted between SAE and proper orthogonal decomposition regarding dimensionality reduction, and the performance of LSTM in time series forecasting is also evaluated against dynamic mode decomposition, where two different training strategies are applied for LSTM within the reduced-order latent space. The proposed SAE-LSTM-based ROM approach is tested on two typical turbulent flow problems for non-intrusive model order reduction. The results demonstrate that the constructed surrogate models possess significant capability in predicting the evolution of turbulent flows by preserving essential nonlinear characteristics inherent in fluid dynamics. This innovative method shows great promise in addressing computational challenges associated with high-resolution numerical modeling applied to complex large-scale flow problems.

## Data
In Case 1, the open-source solver Fluidity is employed to build the computational fluid dynamics (CFD) model utilizing large eddy simulation techniques.
In Case 2, the open-source solver OpenFoam is employed to numerically simulate the turbulent river flow around these two bridge pillars.
The velocity value of snapshots in two cases have been read and save in .pkl and .npy format. You can find them in the './data' directory.

## Install
### Environment
Create a new virtual environment and establish the libraries in this new environment.  
**Conda**  
*Create a new virtual environemnt called 'venv_ROM' with python3 for this project.*
```
conda create -n venv_ROM python=3.6
```
*Enter the conda folder, move vtktools.py to virtual environment path and then install the requirements.txt.*
```
conda activate venv_ROM
```
```
mv dmd_machine/vtktools.py ***/venv_ROM/lib/python3.6/site-packages/   
```
```
pip install -r dmd_machine/requirements.txt  
```

## Modelling
**Case 1: Turbulent flow passing around a circular cylinder**  
Details can be seen in the 'Case1_SAE&LSTM.ipynb

**Case 2: Turbulent river flow passing around multiple bridge pillars**  
Details can be seen in the 'Case2_SAE&LSTM.ipynb'

**Outputs
Plot the result in figure and output the prediction value back to the snapshots. These functions are placed in the './dmd_machine' directory.

## Methodology
More details can be found in the 'Paper.pdf'
[Nonlinear model order reduction of engineering turbulence using data-assisted neural networks]

## License
GNU General Public License v3.0

