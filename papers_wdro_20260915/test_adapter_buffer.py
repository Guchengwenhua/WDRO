"""CPU acceptance probes for the HarRL pilot's physical and gradient interfaces."""
from __future__ import annotations
import json
from pathlib import Path
import math
import numpy as np
import torch
from adapter_buffer import Adapter, ContinuousPhaseTree
from vendor_harrl.evaluator import Node, BufferInsertion, Evaluator


def run():
    torch.set_num_threads(2)
    torch.manual_seed(408)
    model = Adapter(10, 'cpu')
    pins = np.random.default_rng(71).uniform(.1, .9, (6, 10, 2)).astype('float32')
    cost, log_prob, metrics = model.evaluate(pins, False)
    assert cost.shape == log_prob.shape == (6,)
    assert all(np.isfinite(x).all() for x in metrics.values())
    model.actor.bufferloc.log_probabilities[-1].mean().backward()
    last_phase_gradient = float(model.actor.bufferloc.logits.weight.grad.norm())
    assert last_phase_gradient > 0
    assert all(p.grad is None for p in model.actor.start_ptr.parameters())
    model.eval()
    first = model.evaluate(pins, True)[0].detach()
    decisions = model.last_outputs.copy()
    second = model.evaluate(pins, True)[0].detach()
    assert torch.equal(first, second) and np.array_equal(decisions, model.last_outputs)
    one = model.evaluate(pins[:1], False)
    assert one[0].shape == one[1].shape == (1,)

    # Independent Elmore hand calculation for a short two-terminal wire with
    # no harmonic candidate in its interior and fixed endpoint load.
    evaluator = Evaluator(2)
    period = math.sqrt(2 * evaluator.buffer_capacitance * evaluator.buffer_resistance
                       / evaluator.wire_capacitance / evaluator.wire_resistance)
    evaluator.tree = ContinuousPhaseTree(period)
    length = .03
    result = evaluator.eval_func({0: (.1, .1), 1: (.1 + length, .1)},
                                 {0: .5e-15, 1: .5e-15}, [(0, 1, 0.0)])
    expected = evaluator.wire_resistance * length * (.5e-15 + .5 * evaluator.wire_capacitance * length)
    assert math.isclose(result[0], expected, rel_tol=1e-12)
    assert result[3] == 0 and math.isclose(result[2], length, rel_tol=1e-12)

    # A sink may be an internal node: its child's wire and sink loads count.
    parent = Node(1, sink_capacitance=.5)
    parent.add_child(Node(2, capacitance=.25, sink_capacitance=.75))
    capacitance = BufferInsertion(parent, 0, .1, .1).compute_capacitance(parent)
    assert capacitance == 1.5

    tree = ContinuousPhaseTree(1.0)
    coords, segments = tree.insert_steiner(set(), {0: (0., 0.), 1: (1., 0.), 2: (.5, 0.)}, [(0, 1, 0.)])
    assert math.isclose(segments[0][2], 0.) and math.isclose(segments[1][2], math.pi)
    # Equal and opposite waves have zero amplitude, no arbitrary peak.
    coords, edges = tree.find_buffer_positions({0: (0., 0.), 1: (1., 0.)}, [[0, 1, 0., math.pi]], 1.)
    assert len(coords) == 2 and edges == [(0, 1)]

    report = {'status': 'pass', 'tests': ['finite_batched_cost', 'last_phase_gradient',
        'fixed_source_has_no_policy_gradient', 'greedy_determinism', 'single_instance_shape',
        'analytic_two_pin_elmore', 'internal_sink_downstream_capacitance',
        'phase_continuity_across_split', 'harmonic_cancellation'],
        'last_phase_gradient_norm': last_phase_gradient,
        'two_pin_expected_delay': expected, 'two_pin_computed_delay': result[0]}
    Path(__file__).with_name('buffer_cpu_acceptance.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    run()
