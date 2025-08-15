
# awesome PMLTSVM



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)


This repository contains a Python implementation of PMLTSVM, including most of the critical core code. The author plans to organize and upload the complete content to this repository in the future. Below is the directory structure of the code. Datasets downloaded from the mulab library or designed in other papers can be processed using the format conversion tools in the `utils` folder and placed in the `dataset` folder.

```
=================================================
├─ 📁dataset
│  └─ 📄Readme.txt
├─ 📁twsvmlib
│  ├─ 📁__pycache__
│  │  ├─ 📄fastPlane1.cpython-310.pyc
│  │  ├─ 📄fastPlane2.cpython-310.pyc
│  │  ├─ 📄KernelMatrix.cpython-310.pyc
│  │  ├─ 📄metrics.cpython-310.pyc
│  │  ├─ 📄MLTSVM_p.cpython-310.pyc
│  │  ├─ 📄Read.cpython-310.pyc
│  │  ├─ 📄TsvmPlane1.cpython-310.pyc
│  │  ├─ 📄TsvmPlane2.cpython-310.pyc
│  │  ├─ 📄TwinSvm.cpython-310.pyc
│  │  ├─ 📄zfastPlane1.cpython-310.pyc
│  │  ├─ 📄zfastPlane2.cpython-310.pyc
│  │  └─ 📄__init__.cpython-310.pyc
│  ├─ 📄fastPlane1.py
│  ├─ 📄fastPlane2.py
│  ├─ 📄KernelMatrix.py
│  ├─ 📄metrics.py
│  ├─ 📄MLTSVM_k.py
│  ├─ 📄MLTSVM_p.py
│  ├─ 📄Read.py
│  ├─ 📄TsvmPlane1.py
│  ├─ 📄TsvmPlane2.py
│  ├─ 📄TwinSvm.py
│  ├─ 📄zfastPlane1.py
│  ├─ 📄zfastPlane2.py
│  └─ 📄__init__.py
├─ 📁utils
│  ├─ 📄arfftocsv.py
│  ├─ 📄mattocsv.py
│  └─ 📄medical copy.txt
├─ 📄main.py
└─ 📄README.md
```

# Notes

- The project adopts software engineering principles and is designed strictly based on object-oriented programming concepts. The code contains numerous classes and functions, ensuring high encapsulation and non-exposure of internal interfaces. Due to the extensive file structure, it is recommended to first read the `MLTSVM_p.py` file in the `twsvmlib` folder to understand the basic implementation of MLTSVM.
- Be mindful of file paths on different systems. It is recommended to use relative paths instead of absolute paths. The code was developed on macOS, so if you encounter path issues on Linux or Windows, consider modifying the path expressions in `Read.py`. Using the `os` module to handle paths based on your system is advised.
- The `twsvmlib` folder contains the core code, including the implementation of MLTSVM and some auxiliary functions. For large datasets, it is recommended to store them in sparse format and set the `cvxoptimize` solver to `OSQP` to significantly improve computation speed. Relevant updates will be included in future improvements. (Reproducing the results of the paper requires following the experimental setup described in the paper.)
- Some interfaces in the `twsvmlib` code are used for testing and can be modified or removed as needed. For example, the `v` parameter in `def solve(R, S, c1, Epsi1, v=None):` was used for debugging. Due to time constraints, these have not been removed, but you can delete them as needed.
- The `utils` folder contains tools for dataset conversion, primarily for converting datasets in ARFF format to CSV format, making them easier to process with libraries like pandas. You can modify these tools as needed.

# Function Calls

- The code is designed with an interface style similar to the popular scikit-learn library in machine learning. If you are familiar with scikit-learn, you will find the interface design in `twsvmlib` very similar, making it easy to call the methods provided. Below is a simple pipeline for parameter search:

```py
import numpy as np
import Read
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm
from twsvmlib import MLTSVM_p as P
from twsvmlib import metrics as M

if __name__ == "__main__":
    data_list = ['flags']
    folds = 10
    c_vals = [2**i for i in range(-3, 3)]
    g_vals = [2**i for i in range(-3, 3)]  # Adjust parameter range as needed

    for data in tqdm(data_list, desc="Processing"):
        X, y = Read.read(data)
        results = {"Hamming Loss": [], "Ranking Loss": [], "One-Error": [], "Coverage": [], "Avg Precision": [], "Recall": [], "Precision": [], "F1": [], "Total": [], "Best Params": []}
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        for train_idx, test_idx in tqdm(kf.split(X), desc=f"{data} CV", leave=False, total=folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            best_score = float('inf')
            best_params = None

            for c in tqdm(c_vals, desc="Grid Search", leave=False):
                for g in g_vals:
                    model = P.pMLTSVM(c1=c, kernel='rbf', gamma=g)
                    model.fit(X_train, y_train)
                    y_score = model.score(X_val)
                    score = # M.hamming_loss(y_val, y_score), # Choose based on your needs
                    if score < best_score:
                        best_score = score
                        best_params = {'c': c, 'gamma': g}

            results["Best Params"].append(best_params)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X_test)
            final_model = P.pMLTSVM(c1=best_params['c'], kernel='rbf', gamma=best_params['gamma'])
            final_model.fit(X_train, y[train_idx])
            y_pred = final_model.predict(X_test)
            y_score = final_model.score(X_test)
            results["Hamming Loss"].append(M.hamming_loss(y_test, y_pred))
            results["Ranking Loss"].append(M.ranking_loss(y_test, y_score))
            results["One-Error"].append(M.one_error(y_test, y_score))
            results["Coverage"].append(M.coverage(y_test, y_score))
            results["Avg Precision"].append(M.average_precision(y_test, y_score))
            results["Recall"].append(M.recall(y_test, y_pred))
            results["Precision"].append(M.precision(y_test, y_pred))
            results["F1"].append(M.f1_score(y_test, y_pred))
            results["Total"].append(results["Hamming Loss"][-1] + results["Ranking Loss"][-1] + results["One-Error"][-1] + results["Coverage"][-1])

        with open(f"{data}_results.txt", 'w') as f:
            for k, v in results.items():
                if k == "Best Params":
                    continue
                vals = np.array(v)
                f.write(f"{k}: {vals.mean():.4f} ± {vals.std():.4f}\n")
            f.write("\nBest Params:\n")
            for i, p in enumerate(results["Best Params"]):
                f.write(f"Fold {i+1}: c={p['c']:.2e}, gamma={p['gamma']:.2e}\n")
```


