import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(dataset, task=1):
    # task only relevant for Indoorloc (1) and telemonitoring (2), ignored for other datasets
    try:
        X = np.load(os.path.join("datasets", dataset, "X.npy"))
        if dataset in ['Indoorloc', 'telemonitoring']:
            y = np.load(os.path.join("datasets", dataset, f"y{task}.npy"))
        else:
            y = np.load(os.path.join("datasets", dataset, f"y.npy"))
    except FileNotFoundError:
        print("dataset does not exist, please check for available choices in datasets directory")
    assert len(X) == len(y)
    return X, y


def process_data_scale_train(X, y, random_state=None):
    """
    standardize X and y values and performs train test split
    """

    y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled.reshape(-1), train_size=5000,
                                                        random_state=random_state)

    # scale X values. Scaling is fitted only on train data and used to transform both train and test data to avoid
    # data leakage
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def process_data(X, y, random_state=None):
    """
    standardize X and y values and performs train test split
    """

    y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
    X_scaled = preprocessing.StandardScaler().fit_transform(X)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled.reshape(-1), train_size=5000,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def convert_to_pytorch(X_train, X_test, y_train, y_test, batch_size=16, valid_required=True, valid_ratio=0.1,
                       random_state=None):
    """
    builds train, validation and test pytorch dataloaders
    """

    if valid_required:
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, train_size=valid_ratio,
                                                            random_state=random_state)
        valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False)

    else:
        valid_loader = None

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader
