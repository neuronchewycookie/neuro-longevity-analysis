import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_path = Path(__file__).resolve().parents[1] / "scripts" / "08_human_cud.py"
_spec = importlib.util.spec_from_file_location("human_cud_script", _path)
human_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(human_script)


def test_cud_coefficient_uses_control_as_reference():
    # CUD donors score 1.0 higher; the CUD coefficient must come out positive (~ +1), not flipped
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        dx = "CUD" if i % 2 else "Control"
        for region in ("Caudate", "NAc"):
            rows.append({"subject_id": f"S{i}", "region": region, "diagnosis": dx,
                         "age": 20 + i % 30, "pmi": 10 + i % 15, "race": "White" if i % 3 else "Black",
                         "cas": (1.0 if dx == "CUD" else 0.0) + rng.normal(0, 0.3)})
    models = pd.DataFrame(human_script.fit_models(pd.DataFrame(rows), "test"))
    assert (models["coef_cud"].between(0.7, 1.3)).all()
