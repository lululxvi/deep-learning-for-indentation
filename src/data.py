from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd


class FEMData(object):
    def __init__(self, yname, angles):
        self.yname = yname
        self.angles = angles

        self.X = None
        self.y = None

        if len(angles) == 1:
            self.read_1angle()
        elif len(angles) == 2:
            self.read_2angles()
        elif len(angles) == 4:
            self.read_4angles()

    def read_1angle(self):
        df = pd.read_csv("../data/FEM_{}deg.csv".format(self.angles[0]))
        df["E* (GPa)"] = EtoEstar(df["E (GPa)"])
        df["sy/E*"] = df["sy (GPa)"] / df["E* (GPa)"]
        df = df.loc[~((df["n"] > 0.3) & (df["sy/E*"] >= 0.03))]
        # df = df.loc[df["n"] <= 0.3]
        # Scale c* from Conical to Berkovich
        # df["dP/dh (N/m)"] *= 1.167 / 1.128
        # Add noise
        # sigma = 0.2
        # df["E* (GPa)"] *= 1 + sigma * np.random.randn(len(df))
        # df["sy (GPa)"] *= 1 + sigma * np.random.randn(len(df))
        print(df.describe())

        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt"]].values
        if self.yname == "E*":
            self.y = df["E* (GPa)"].values[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        elif self.yname.startswith("sigma_"):
            e_plastic = float(self.yname[6:])
            self.y = (
                df["sy (GPa)"]
                * (1 + e_plastic * df["E (GPa)"] / df["sy (GPa)"]) ** df["n"]
            ).values[:, None]

    def read_2angles(self):
        df1 = pd.read_csv("../data/FEM_70deg.csv")
        df2 = pd.read_csv("../data/FEM_60deg.csv")
        df = df1.set_index("Case").join(
            df2.set_index("Case"), how="inner", rsuffix="_60"
        )
        # df = df.loc[:100]
        print(df.describe())

        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt", "C (GPa)_60"]].values
        # self.X = df[["C (GPa)", "dP/dh (N/m)", "C (GPa)_60", "dP/dh (N/m)_60"]].values
        if self.yname == "E*":
            self.y = EtoEstar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]

    def read_4angles(self):
        df1 = pd.read_csv("../data/FEM_50deg.csv")
        df2 = pd.read_csv("../data/FEM_60deg.csv")
        df3 = pd.read_csv("../data/FEM_70deg.csv")
        df4 = pd.read_csv("../data/FEM_80deg.csv")
        df = (
            df3.set_index("Case")
            .join(df1.set_index("Case"), how="inner", rsuffix="_50")
            .join(df2.set_index("Case"), how="inner", rsuffix="_60")
            .join(df4.set_index("Case"), how="inner", rsuffix="_80")
        )
        print(df.describe())

        self.X = df[
            [
                "C (GPa)",
                "dP/dh (N/m)",
                "Wp/Wt",
                "C (GPa)_50",
                "C (GPa)_60",
                "C (GPa)_80",
            ]
        ].values
        if self.yname == "E*":
            self.y = EtoEstar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]


class ModelData(object):
    def __init__(self, yname, n, model):
        self.yname = yname
        self.n = n
        self.model = model

        self.X = None
        self.y = None

        self.read()

    def read(self):
        df = pd.read_csv("../data/model_{}.csv".format(self.model))
        self.X = df[["C (GPa)", "dP/dh (N/m)", "WpWt"]].values
        if self.yname == "E*":
            self.y = EtoEstar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        idx = np.random.choice(np.arange(len(self.X)), self.n, replace=False)
        self.X = self.X[idx]
        self.y = self.y[idx]


class ExpData(object):
    def __init__(self, filename, yname):
        self.filename = filename
        self.yname = yname

        self.X = None
        self.y = None

        self.read()

    def read(self):
        df = pd.read_csv(self.filename)
        # Scale dP/dh from 3N to hm = 0.2um
        # df["dP/dh (N/m)"] *= 0.2 * (df["C (GPa)"] / 3) ** 0.5 * 10 ** (-1.5)
        # Scale dP/dh from Pm to hm = 0.2um
        # df["dP/dh (N/m)"] *= 0.2 * (df["C (GPa)"] / df["Pm (N)"]) ** 0.5 * 10 ** (-1.5)
        # Scale dP/dh from hm to hm = 0.2um
        # df["dP/dh (N/m)"] *= 0.2 / df["hm (um)"]
        # Scale c* from Berkovich to Conical
        # df["dP/dh (N/m)"] *= 1.128 / 1.167
        print(df.describe())

        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt"]].values
        if self.yname == "E*":
            self.y = df["E* (GPa)"].values[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        elif self.yname.startswith("sigma_"):
            e_plastic = self.yname[6:]
            self.y = df["s" + e_plastic + " (GPa)"].values[:, None]


class BerkovichData(object):
    def __init__(self, yname, scale_c=False):
        self.yname = yname
        self.scale_c = scale_c

        self.X = None
        self.y = None

        self.read()

    def read(self):
        df = pd.read_csv("../data/Berkovich.csv")
        # Scale c* from Berkovich to Conical
        if self.scale_c:
            df["dP/dh (N/m)"] *= 1.128 / 1.167
        print(df.describe())

        self.X = df[["C (GPa)", "dP/dh (N/m)", "Wp/Wt"]].values
        if self.yname == "E*":
            self.y = EtoEstar(df["E (GPa)"].values)[:, None]
        elif self.yname == "sigma_y":
            self.y = df["sy (GPa)"].values[:, None]
        elif self.yname == "n":
            self.y = df["n"].values[:, None]
        elif self.yname.startswith("sigma_"):
            e_plastic = float(self.yname[6:])
            self.y = (
                df["sy (GPa)"]
                * (1 + e_plastic * df["E (GPa)"] / df["sy (GPa)"]) ** df["n"]
            ).values[:, None]


def EtoEstar(E):
    nu = 0.3
    nu_i, E_i = 0.07, 1100
    return 1 / ((1 - nu ** 2) / E + (1 - nu_i ** 2) / E_i)
