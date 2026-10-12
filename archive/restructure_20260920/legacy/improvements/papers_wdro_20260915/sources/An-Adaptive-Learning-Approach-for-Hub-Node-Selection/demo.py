"""Small end-to-end example without pretrained weights."""

import numpy as np
import torch

from models.evaluate import evaluate_batch
from models.selector_conductor import Actor


def main() -> None:
    degree = 10
    points = np.random.default_rng(9).random((2, degree, 2))
    actor = Actor(degree, torch.device("cpu"))
    actor.eval()
    with torch.no_grad():
        hubs, _, _ = actor(points, steps=5, deterministic=True)
    wirelength, skew, objective = evaluate_batch(points, hubs.numpy(), objective=2)
    print("hub indices:\n", hubs.numpy())
    print("wirelength:", wirelength)
    print("skew:", skew)
    print("objective:", objective)


if __name__ == "__main__":
    main()
