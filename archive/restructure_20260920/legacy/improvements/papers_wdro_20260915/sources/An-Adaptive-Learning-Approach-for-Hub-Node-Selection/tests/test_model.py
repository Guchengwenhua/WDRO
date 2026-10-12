import numpy as np
import torch

from models.selector_conductor import Actor, Critic


def test_actor_and_critic_shapes():
    degree = 8
    points = np.random.default_rng(0).random((3, degree, 2))
    actor = Actor(degree, torch.device("cpu"))
    critic = Critic(degree, torch.device("cpu"))
    actor.eval()
    critic.eval()
    with torch.no_grad():
        hubs, log_prob, reward = actor(points, steps=3, deterministic=True)
        prediction = critic(points)
    assert hubs.shape == (3, 3)
    assert log_prob.shape == (3,)
    assert reward.shape == (3,)
    assert prediction.shape == (3,)
    assert torch.all((hubs >= 0) & (hubs < degree))
