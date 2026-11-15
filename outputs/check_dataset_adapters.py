import json
from pathlib import Path

from experiments.multidataset_benchmark import prepare_split
from pwgp.data.benchmarks import load_real, synthetic_dataset


report = {}
for name in ("california", "protein", "sarcos", "tco", "tco_l2", "icethick"):
    dataset = load_real(name, cache="data/multidataset_sources_20261112",
                        input_path="C:/Users/ADMIN/Downloads/icethk.txt" if name == "icethick" else None)
    split = prepare_split(dataset, train_fraction=.8, seed=31)
    report[name] = {"rows": len(dataset.y), "features": dataset.x.shape[1],
                    "train": len(split["train"]), "test": len(split["test"]),
                    "target_std": split["target_std"], "metadata": dataset.metadata}
    print(name, report[name], flush=True)
for function in ("michalewicz", "g_function"):
    for dimension in (3, 5, 10):
        for count in (50000, 100000):
            dataset = synthetic_dataset(function, dimension=dimension, training_points=count,
                test_points=count // 4, seed=31, noise_variance=1e-8, michalewicz_m=10,
                g_coefficients=[i / 2 for i in range(dimension)])
            split = prepare_split(dataset, train_fraction=.8, seed=31)
            assert len(split["train"]) == count and len(split["test"]) == count // 4
            name = f"{function}_d{dimension}_n{count}"
            report[name] = {"rows": len(dataset.y), "features": dimension,
                            "train": count, "test": count // 4, "target_std": split["target_std"]}
            print(name, report[name], flush=True)
Path("data/multidataset_sources_20261112/preparation_checks.json").write_text(json.dumps(report, indent=2))
