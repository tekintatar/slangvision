# tests/conftest.py — marks the CUDA tests so `pytest -m "not gpu"` runs the
# GPU-free remainder (CI, machines without an NVIDIA GPU).

import pytest

GPU_TEST_FILES = {"test_pipelines.py"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name in GPU_TEST_FILES:
            item.add_marker(pytest.mark.gpu)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped = [r for r in terminalreporter.stats.get("skipped", [])
               if GPU_TEST_FILES & {str(r.nodeid).split("::")[0].rsplit("/", 1)[-1]}]
    if skipped:
        terminalreporter.write_line(
            "CUDA tests were skipped (no torch, torchvision or CUDA GPU): "
            "this run did not validate the kernels.")
