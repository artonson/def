import math
import unittest
import random

from bridson import poisson_disc_samples


class PoissonDiscSamplingRadiusTest(unittest.TestCase):
    def setUp(self):
        self.prng = random.Random()
        self.prng.seed(42)

    def test_r10(self):
        r = 10
        samples = poisson_disc_samples(100, 100, r=r, random=self.prng.random)
        self.assertGreater(len(samples), 50)
        for i, p in enumerate(samples):
            for q in samples[i + 1:]:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                self.assertGreater(math.sqrt(dx * dx + dy * dy), r)

    def test_r5(self):
        r = 5
        samples = poisson_disc_samples(100, 100, r=r, random=self.prng.random)
        self.assertGreater(len(samples), 200)
        for i, p in enumerate(samples):
            for q in samples[i + 1:]:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                self.assertGreater(math.sqrt(dx * dx + dy * dy), r)

    def test_r50(self):
        r = 50
        samples = poisson_disc_samples(100, 100, r=r, random=self.prng.random)
        self.assertGreater(len(samples), 2)
        for i, p in enumerate(samples):
            for q in samples[i + 1:]:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                self.assertGreater(math.sqrt(dx * dx + dy * dy), r)
