import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_scenario3_linear_baseline import run
res = run(["linear_gate_moreno", "sedenion", "linear_gate_moreno_noffn", "sedenion_noffn"], epochs=12, n_train=24000)
(pathlib.Path(__file__).resolve().parent / "results_scenario3_converged.json").write_text(json.dumps(res, indent=1))
