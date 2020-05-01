from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, ShuffleSplit

import deepxde as dde
from data import BerkovichData, ExpData, FEMData, ModelData


def svm(data):
    clf = SVR(kernel="rbf")
    clf.fit(data.train_x, data.train_y[:, 0])
    y_pred = clf.predict(data.test_x)[:, None]
    return dde.metrics.get("MAPE")(data.test_y, y_pred)


def mfgp(data):
    from mfgp import LinearMFGP

    model = LinearMFGP(noise=0, n_optimization_restarts=5)
    model.train(data.X_lo_train, data.y_lo_train, data.X_hi_train, data.y_hi_train)
    _, _, y_pred, _ = model.predict(data.X_hi_test)
    return dde.metrics.get("MAPE")(data.y_hi_test, y_pred)


def nn(data):
    layer_size = [data.train_x.shape[1]] + [32] * 2 + [1]
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]

    loss = "MAPE"
    optimizer = "adam"
    if data.train_x.shape[1] == 3:
        lr = 0.0001
    else:
        lr = 0.001
    epochs = 30000

    net = dde.maps.FNN(
        layer_size, activation, initializer, regularization=regularization
    )
    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, loss=loss, metrics=["MAPE"])
    losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return train_state.best_metrics[0]


def validation_model(yname, train_size):
    datafem = FEMData(yname, [70])

    mape = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))

        datamodel = ModelData(yname, train_size, "forward")
        X_train, X_test = datamodel.X, datafem.X
        y_train, y_test = datamodel.y, datafem.y

        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        # mape.append(svm(data))
        mape.append(nn(data))

    print(yname, train_size)
    print(np.mean(mape), np.std(mape))


def validation_FEM(yname, angles, train_size):
    datafem = FEMData(yname, angles)
    # datafem = BerkovichData(yname)

    if train_size == 80:
        kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)
    elif train_size == 90:
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
    else:
        kf = ShuffleSplit(
            n_splits=10, test_size=len(datafem.X) - train_size, random_state=0
        )

    mape = []
    iter = 0
    for train_index, test_index in kf.split(datafem.X):
        iter += 1
        print("\nCross-validation iteration: {}".format(iter))

        X_train, X_test = datafem.X[train_index], datafem.X[test_index]
        y_train, y_test = datafem.y[train_index], datafem.y[test_index]

        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        mape.append(dde.apply(nn, (data,)))

    print(mape)
    print(yname, train_size, np.mean(mape), np.std(mape))


def mfnn(data):
    x_dim, y_dim = 3, 1
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]
    net = dde.maps.MfNN(
        [x_dim] + [128] * 2 + [y_dim],
        [8] * 2 + [y_dim],
        activation,
        initializer,
        regularization=regularization,
        residue=True,
        trainable_low_fidelity=True,
        trainable_high_fidelity=True,
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001, loss="MAPE", metrics=["MAPE", "APE SD"])
    losshistory, train_state = model.train(epochs=30000)
    # checker = dde.callbacks.ModelCheckpoint(
    #     "model/model.ckpt", verbose=1, save_better_only=True, period=1000
    # )
    # losshistory, train_state = model.train(epochs=30000, callbacks=[checker])
    # losshistory, train_state = model.train(epochs=5000, model_restore_path="model/model.ckpt-28000")

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return (
        train_state.best_metrics[1],
        train_state.best_metrics[3],
        train_state.best_y[1],
    )


def validation_mf(yname, train_size):
    datalow = FEMData(yname, [70])
    # datalow = ModelData(yname, 10000, "forward_n")
    datahigh = BerkovichData(yname)
    # datahigh = FEMData(yname, [70])

    kf = ShuffleSplit(
        n_splits=10, test_size=len(datahigh.X) - train_size, random_state=0
    )
    # kf = LeaveOneOut()

    mape = []
    iter = 0
    for train_index, test_index in kf.split(datahigh.X):
        iter += 1
        print("\nCross-validation iteration: {}".format(iter), flush=True)

        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=datahigh.X[train_index],
            y_lo_train=datalow.y,
            y_hi_train=datahigh.y[train_index],
            X_hi_test=datahigh.X[test_index],
            y_hi_test=datahigh.y[test_index],
        )
        mape.append(dde.apply(mfnn, (data,))[0])
        # mape.append(dde.apply(mfgp, (data,)))

    print(mape)
    print(yname, train_size, np.mean(mape), np.std(mape))


def validation_scaling(yname):
    datafem = FEMData(yname, [70])
    # dataexp = ExpData(yname)
    dataexp = BerkovichData(yname, scale_c=True)

    mape = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))
        data = dde.data.DataSet(
            X_train=datafem.X, y_train=datafem.y, X_test=dataexp.X, y_test=dataexp.y
        )
        mape.append(nn(data))

    print(yname)
    print(np.mean(mape), np.std(mape))


def validation_exp(yname):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/B3067.csv", yname)

    ape = []
    y = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=dataBerkovich.X,
            y_lo_train=datalow.y,
            y_hi_train=dataBerkovich.y,
            X_hi_test=dataexp.X,
            y_hi_test=dataexp.y,
        )
        res = dde.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross(yname):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/B3067.csv", yname)
    train_size = 10

    ape = []
    y = []

    # cases = range(6)
    # for train_index in itertools.combinations(cases, 3):
    #     train_index = list(train_index)
    #     test_index = list(set(cases) - set(train_index))

    kf = ShuffleSplit(
        n_splits=10, test_size=len(dataexp.X) - train_size, random_state=0
    )
    for train_index, test_index in kf.split(dataexp.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index, "==>", test_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp.X[train_index])),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp.y[train_index])),
            X_hi_test=dataexp.X[test_index],
            y_hi_test=dataexp.y[test_index],
        )
        res = dde.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross2(yname, train_size):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp1 = ExpData("../data/B3067.csv", yname)
    dataexp2 = ExpData("../data/B3090.csv", yname)

    ape = []
    y = []

    kf = ShuffleSplit(n_splits=10, train_size=train_size, random_state=0)
    for train_index, _ in kf.split(dataexp1.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp1.X[train_index])),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp1.y[train_index])),
            X_hi_test=dataexp2.X,
            y_hi_test=dataexp2.y,
        )
        res = dde.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname, train_size)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross3(yname):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp1 = ExpData("../data/Al6061.csv", yname)
    dataexp2 = ExpData("../data/Al7075.csv", yname)

    ape = []
    y = []
    for _ in range(10):
        print("\nIteration: {}".format(len(ape)))
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp1.X)),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp1.y)),
            X_hi_test=dataexp2.X,
            y_hi_test=dataexp2.y,
        )
        res = dde.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt("y.dat", np.hstack(y))


def validation_exp_cross_transfer(yname):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/B3090.csv", yname)
    train_size = 5

    data = dde.data.MfDataSet(
        X_lo_train=datalow.X,
        X_hi_train=dataBerkovich.X,
        y_lo_train=datalow.y,
        y_hi_train=dataBerkovich.y,
        X_hi_test=dataexp.X,
        y_hi_test=dataexp.y,
    )
    res = dde.apply(mfnn, (data,))
    return

    ape = []
    y = []

    # cases = range(6)
    # for train_index in itertools.combinations(cases, 3):
    #     train_index = list(train_index)
    #     test_index = list(set(cases) - set(train_index))

    kf = ShuffleSplit(
        n_splits=10, test_size=len(dataexp.X) - train_size, random_state=0
    )
    for train_index, test_index in kf.split(dataexp.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index, "==>", test_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=dataexp.X[train_index],
            y_lo_train=datalow.y,
            y_hi_train=dataexp.y[train_index],
            X_hi_test=dataexp.X[test_index],
            y_hi_test=dataexp.y[test_index],
        )
        res = dde.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)


def main():
    # validation_FEM("E*", [50, 60, 70, 80], 70)
    # validation_mf("E*", 9)
    # validation_scaling("E*")
    # validation_exp("E*")
    # validation_exp_cross("E*")
    # validation_exp_cross2("E*", 10)
    # validation_exp_cross3("E*")
    # validation_exp_cross_transfer("E*")
    # return

    for train_size in range(1, 10):
        # validation_model("E*", train_size)
        # validation_FEM("sigma_y", [50, 60, 70, 80], train_size)
        # validation_mf("E*", train_size)
        validation_exp_cross2("E*", train_size)
        print("=======================================================")
        print("=======================================================")


if __name__ == "__main__":
    main()
