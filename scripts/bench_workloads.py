import time
import sqlite3
import tempfile
import os
import numpy  as np
import pandas as pd
from sklearn.ensemble     import RandomForestClassifier
from sklearn.datasets     import make_classification

N_REPEAT    = 3
RANDOM_SEED = 42

def _time_it(func, *args) -> float:
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return min(times)

def _sort_numpy():
    rng = np.random.default_rng(RANDOM_SEED)
    arr = rng.random(5_000_000)
    np.sort(arr)

def _sort_python():
    rng = np.random.default_rng(RANDOM_SEED)
    lst = rng.integers(0, 10_000_000, size=1_000_000).tolist()
    sorted(lst)


def _join_pandas():

    rng = np.random.default_rng(RANDOM_SEED)
    n   = 500_000

    orders = pd.DataFrame({
        "order_id":    np.arange(n),
        "customer_id": rng.integers(0, n // 2, size=n),
        "amount":      rng.random(n) * 1000,
    })
    customers = pd.DataFrame({
        "customer_id": np.arange(n // 2),
        "age":         rng.integers(18, 80, size=n // 2),
        "score":       rng.random(n // 2),
    })
    result = orders.merge(customers, on="customer_id", how="inner")
    return len(result)


def _ml_train():

    X, y = make_classification(
        n_samples=50_000, n_features=20, n_informative=15,
        random_state=RANDOM_SEED
    )
    model = RandomForestClassifier(
        n_estimators=50, max_depth=10,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    model.fit(X, y)
    return model

def _ml_inference():
    X, y = make_classification(
        n_samples=60_000, n_features=20, n_informative=15,
        random_state=RANDOM_SEED
    )
    X_train, X_test = X[:50_000], X[50_000:]
    y_train         = y[:50_000]

    model = RandomForestClassifier(
        n_estimators=50, max_depth=10,
        n_jobs=-1, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    t0   = time.perf_counter()
    preds = model.predict(X_test)
    return time.perf_counter() - t0

def _sql_benchmark():

    rng = np.random.default_rng(RANDOM_SEED)
    n   = 100_000
    conn = sqlite3.connect(":memory:")
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE sales (
            id       INTEGER PRIMARY KEY,
            region   TEXT,
            product  TEXT,
            amount   REAL,
            quantity INTEGER
        )
    """)

    regions  = ["Nord", "Sud", "Est", "Vest", "Centru"]
    products = ["Laptop", "Telefon", "Tableta", "Monitor", "Tastatura"]

    rows = [
        (
            i,
            regions [rng.integers(0, len(regions))],
            products[rng.integers(0, len(products))],
            float(rng.random() * 5000),
            int(rng.integers(1, 100)),
        )
        for i in range(n)
    ]
    cur.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?)", rows
    )
    conn.commit()

    cur.execute("SELECT * FROM sales WHERE amount > 2500")
    cur.fetchall()

    cur.execute("""
        SELECT region, COUNT(*) as total, AVG(amount) as avg_amount
        FROM sales
        GROUP BY region
        ORDER BY avg_amount DESC
    """)
    cur.fetchall()

    conn.close()

def run_workload_benchmarks() -> dict:
    results = {}

    results["sort_numpy_s"]      = _time_it(_sort_numpy)
    results["sort_python_s"]     = _time_it(_sort_python)
    results["join_pandas_s"]     = _time_it(_join_pandas)

    t0 = time.perf_counter()
    _ml_train()
    results["ml_train_s"]        = time.perf_counter() - t0

    results["ml_inference_s"]    = _ml_inference()
    results["sql_benchmark_s"]   = _time_it(_sql_benchmark)

    return results
