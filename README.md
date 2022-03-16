# Extraction of mechanical properties of materials through deep learning from instrumented indentation

The data and code for the paper [L. Lu, M. Dao, P. Kumar, U. Ramamurty, G. E. Karniadakis, & S. Suresh. Extraction of mechanical properties of materials through deep learning from instrumented indentation. *Proceedings of the National Academy of Sciences*, 117(13), 7052-7062, 2020](https://www.pnas.org/content/early/2020/03/13/1922210117).

## Data

All the data is in the folder [data](data).

## Code

All the code is in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v1.1.2. If you use DeepXDE>1.1.2, you need to set `standardize=True` in `dde.data.MfDataSet()`.

- [data.py](src/data.py): The classes are used to read the data file. Remember to uncomment certain line in `ExpData` to scale `dP/dh`.
- [nn.py](src/nn.py): The main functions of multi-fidelity neural networks.
- [model.py](src/model.py): The fitting function method. Some parameters are hard-coded in the code, and you should modify them for different cases.
- [fit_n.py](src/fit_n.py): Fit strain-hardening exponent.
- [mfgp.py](src/mfgp.py): Multi-fidelity Gaussian process regression.

## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{Lu7052,
  author  = {Lu, Lu and Dao, Ming and Kumar, Punit and Ramamurty, Upadrasta and Karniadakis, George Em and Suresh, Subra},
  title   = {Extraction of mechanical properties of materials through deep learning from instrumented indentation},
  volume  = {117},
  number  = {13},
  pages   = {7052--7062},
  year    = {2020},
  doi     = {10.1073/pnas.1922210117},
  journal = {Proceedings of the National Academy of Sciences}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
