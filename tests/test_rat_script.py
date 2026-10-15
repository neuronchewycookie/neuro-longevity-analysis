import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_path = Path(__file__).resolve().parents[1] / "scripts" / "07_rat_reward_sex.py"
_spec = importlib.util.spec_from_file_location("rat_reward_sex", _path)
rat_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rat_script)


def test_control_standardized_uses_own_file_control():
    scores = pd.DataFrame({
        "file": ["a"] * 4 + ["b"] * 4,
        "group": ["Saline", "Saline", "Morphine_D1", "Morphine_D30", "Cue", "Cue", "Sucrose_D1", "Sucrose_D30"],
        "time_point": [None, None, "Day 1", "Day 30", None, None, "Day 1", "Day 30"],
        "reinforcer": ["Morphine"] * 4 + ["Sucrose"] * 4,
        # file b is shifted by +100: standardizing within file must remove the offset
        "cas": [0.0, 2.0, 3.0, 1.0, 100.0, 102.0, 103.0, 101.0],
    })
    out = rat_script.control_standardized(scores).set_index("group")["delta_z"]
    sd = np.std([0.0, 2.0], ddof=1)
    assert np.isclose(out["Morphine_D1"], 2 / sd)
    assert np.isclose(out["Sucrose_D1"], out["Morphine_D1"])
    assert set(out.index) == {"Morphine_D1", "Morphine_D30", "Sucrose_D1", "Sucrose_D30"}
