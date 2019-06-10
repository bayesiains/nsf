"""Tests for the 2-dim plane datasets."""

import torchtestcase
import unittest

from data import plane


class PlaneDatasetTest(torchtestcase.TorchTestCase):

    def test_all(self):
        num_points = 40
        constructors = [
            plane.GaussianDataset,
            plane.CrescentDataset,
            plane.CrescentCubedDataset,
            plane.SineWaveDataset,
            plane.AbsDataset,
            plane.SignDataset,
            plane.FourCircles,
            plane.DiamondDataset,
            plane.TwoSpiralsDataset,
            plane.CheckerboardDataset,
        ]
        for constructor in constructors:
            for flip_axes in [True, False]:
                with self.subTest(constructor=constructor, flip_axes=flip_axes):
                    dataset = constructor(num_points=num_points, flip_axes=flip_axes)
                    dataset.reset()
                    self.assertEqual(len(dataset), num_points)
                    self.assertEqual(dataset[0], dataset.data[0])


class FaceDatasetTest(torchtestcase.TorchTestCase):

    def test_all(self):
        num_points = 40
        for name in ['einstein', 'boole', 'bayes']:
            for flip_axes in [True, False]:
                with self.subTest(name=name, flip_axes=flip_axes):
                    dataset = plane.FaceDataset(
                        num_points=num_points, flip_axes=flip_axes, name=name)
                    dataset.reset()
                    self.assertEqual(len(dataset), num_points)
                    self.assertEqual(dataset[0], dataset.data[0])


class TestGridDatasetTest(torchtestcase.TorchTestCase):

    def test_all(self):
        num_points = 40
        bounds = [[-1, 1]] * 2
        for flip_axes in [True, False]:
            with self.subTest(flip_axes=flip_axes):
                dataset = plane.TestGridDataset(
                    num_points_per_axis=num_points, bounds=bounds)
                dataset.reset()
                self.assertEqual(len(dataset), num_points ** 2)
                self.assertEqual(dataset[0], dataset.data[0])


if __name__ == '__main__':
    unittest.main()
