# Extraction of mechanical properties of materials through deep learning from instrumented indentation

The data and code for the paper "Extraction of mechanical properties of materials through deep learning from instrumented indentation".

## Data

All the data is in the folder [data](data).

## Code

All the code is in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde).

- [data.py](src/data.py): The classes are used to read the data file. Remember to uncomment certain line in `ExpData` to scale `dP/dh`.
- [nn.py](src/nn.py): The main functions of multi-fidelity neural networks.
- [model.py](src/model.py): The fitting function method. Some parameters are hard-coded in the code, and you should modify them for different cases.
- [fit_n.py](src/fit_n.py): Fit strain-hardening exponent.
- [mfgp.py](src/mfgp.py): Multi-fidelity Gaussian process regression.

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.