"""Export configuration and signature metadata without constructing or training models."""
import ast
import inspect
import json
import sys
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from experiments.default_hyperparameters import heaton_defaults, oisst_defaults, ukhousing_defaults
from experiments.run_selection import select_runs
from pwgp.config import HyperparameterBounds, NumericalConfig, OptimizerConfig
from pwgp.models.nested_kriging import NestedKriging
from pwgp.models.patchwork import GeneralizedPatchworkGP
from pwgp.models.prospar import ProSparGP
from pwgp.models.standard_patchwork import StandardPatchworkGP
from pwgp.models.svgp import SVGP
from pwgp.models.vnngp import VNNGP
from pwgp.training import fit
import torch

def value(v):
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, (str, bool, float, int, tuple, list, dict)) or v is None:
        return v
    return str(v)

def signature(fn):
    return {k: {"default": "REQUIRED" if v.default is inspect.Parameter.empty else value(v.default),
                "kind": str(v.kind)} for k, v in inspect.signature(fn).parameters.items()
            if k != "self" and v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}

classes = {"SVGP": SVGP, "GPW": GeneralizedPatchworkGP, "ProSpar": ProSparGP,
           "VNNGP": VNNGP, "Standard Patchwork": StandardPatchworkGP, "Nested Kriging": NestedKriging}
data, runs = oisst_defaults(device="cuda:0")
runs = select_runs(runs, models=[name for name, _ in runs], overrides={})
presets = [{"dataset": "OISST", "label": label, "config": asdict(config)} for label, config in runs]
for name, model in [("SVGP", "svgp"), ("GPW", "gpw"), ("ProSpar", "prospar"),
                    ("VNNGP", "vnngp"), ("Standard Patchwork (legacy adaptive)", "patchwork")]:
    presets.append({"dataset": "UKHousing", "label": name, "config": asdict(replace(ukhousing_defaults(device="cuda:0"), model=model))})
for name, model in [("SVGP", "svgp"), ("GPW", "patchwork"), ("ProSpar", "prospar")]:
    presets.append({"dataset": "Heaton", "label": name, "config": asdict(replace(heaton_defaults(device="cuda:0"), model=model))})

# The AST inventory checks every dataclass field, including inactive controls.
schema = {}
for file, names in [("experiments/ukhousing_benchmark.py", ["UKConfig"]),
                    ("experiments/heaton_benchmark.py", ["HeatonConfig"]),
                    ("src/pwgp/config.py", ["NumericalConfig", "OptimizerConfig", "HyperparameterBounds", "ProSparConfig"])]:
    for node in ast.parse((root / file).read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            schema[node.name] = [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]
result = {"presets": presets, "data": asdict(data), "schema": schema,
          "constructors": {name: signature(cls.__init__) for name, cls in classes.items()},
          "prediction": {name: signature(cls.predict) for name, cls in classes.items()},
          "fit": signature(fit), "nk_fit": signature(NestedKriging.fit),
          "numerical": asdict(NumericalConfig()), "bounds": asdict(HyperparameterBounds()),
          "optimizer": asdict(OptimizerConfig()), "adam": signature(torch.optim.Adam.__init__)}
destination = Path(__file__).with_name("settings_inventory.json")
destination.write_text(json.dumps(result, indent=2))
print("Extracted", len(presets), "preset configurations and", len(classes), "model APIs.")
