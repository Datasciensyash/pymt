from pathlib import Path

import numpy as np
import pytest
import yaml

from pymt.direct_task import direct_task_1d


@pytest.mark.parametrize(
    "test_case_path", (Path(__file__).parent / "direct_task_test_case.yml",)
)
def test_direct_task(test_case_path: Path):
    with test_case_path.open() as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    rho, phi = direct_task_1d(
        periods=np.array(config["periods"]),
        layer_resistivity=np.array(config["layer_resistivity"]),
        layer_power=np.array(config["layer_power"]),
    )

    rho, phi = np.round(rho, 2), np.round(phi, 2)

    assert np.allclose(phi, np.array(config["phi"]))
    assert np.allclose(rho, np.array(config["rho"]))
