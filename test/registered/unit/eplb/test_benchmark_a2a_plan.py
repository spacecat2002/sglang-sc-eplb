import importlib.util
from pathlib import Path


_PATH = Path(__file__).parents[4] / "benchmark" / "benchmark_a2a_plan.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_a2a_plan", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_physical_maps_pad_layouts_and_select_source_local_replicas():
    layouts = {
        "baseline": {0: (0,), 1: (0,), 2: (1,), 3: (1,)},
        "plan": {0: (1, 0), 1: (0,), 2: (1,), 3: (1,)},
    }

    slots, maps = _MODULE._physical_maps(layouts, num_experts=4, num_ranks=2)

    assert slots == 3
    assert maps["plan"][0][0] // slots == 0
    assert maps["plan"][1][0] // slots == 1
    assert [value // slots for value in maps["baseline"][0]] == [0, 0, 1, 1]
