# WASP43b MIRI retrievals repository
This repository contains the main retrieval results from Bell et al. 2024, Nat.  That is, the results from the Hydra, Pyrat Bay, NEMESIS, PLATON, and SCARLET retrieval frameworks. 
Also, this repository contains Python scripts that allow you to reproduce Figure 4, Extended-data Fig. 6, and Extended-data Fig. 8 of the article.

### Retrieval output data
The `data` folder contains python `.npz` files containing the main retrieval results: temperature profiles (mean, -1sigma, and +1sigma), VMRs for H2O and CH4, and contribution functions.  For example, to access the VMR posterior for Hydra run from a Python session:
```python
import numpy as np
data = np.load('vmr_posterior_hydra.npz')

# Show file content:
for key in data.keys():
    print(key)
parameters
vmr_units
vmr_posterior_phase_0.00
vmr_posterior_phase_0.25
vmr_posterior_phase_0.50
vmr_posterior_phase_0.75

# Get the dayside VMR posterior distribution:
dayside_VMR = data['vmr_posterior_phase_0.50']
# This is a 2D array with shape [nsamples,nmolecules]
print(dayside_VMR.shape)
(12154, 2)

# These are the molecules / free parameters
print(data['parameters'])
['H2O' 'CH4']

# With these units:
print(data['vmr_units'])
log10(VMR)

```

### Requirements to reproduce figures
To run the scripts that reproduce the article's Figures you need Python >= 3.6; and the packages below, with the recommended versions (previous version might probably still work).  The command lines below show you one way to install these packages if you don't have them already in your python environment.

```shell
pip install numpy >= 1.25
pip install matplotlib >= 3.7
pip install mc3 >= 3.1.3
```

### Recreate the plots
To recreate the plots in your computer you can execute the `fig_WASP43b_retrieval*.py` files from the command line.  Make sure that you are located in the folder containing the `.py` file.  

```shell
# For Figure 4:
python fig_WASP43b_retrieval_temp_vmr.py free

# For ED Figure 8:
python fig_WASP43b_retrieval_temp_vmr.py equilibrium

# For ED Figure 6:
python fig_WASP43b_retrieval_contributions.py
```
The output pdf files will be located in the `plots` folder.
