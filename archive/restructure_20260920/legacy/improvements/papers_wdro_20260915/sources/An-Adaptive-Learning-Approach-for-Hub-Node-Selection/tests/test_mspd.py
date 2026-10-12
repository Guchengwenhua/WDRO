import numpy as np

from models.evaluate import evaluate_instance, objective_weights


def test_objective_weights():
    assert objective_weights(1) == (0.5, 0.5)
    assert objective_weights(2) == (0.25, 0.75)
    assert objective_weights(3) == (0.75, 0.25)
    assert objective_weights(0.2) == (0.2, 0.8)


def test_mspd_evaluation_is_finite():
    points = np.array(
        [[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [3, 8]],
        dtype=np.int64,
    )
    wirelength, skew, value = evaluate_instance(
        points, [1, 4], objective=2, coordinate_scale=1
    )
    assert wirelength > 0
    assert skew >= 0
    assert value == 0.25 * wirelength + 0.75 * skew
