import unittest

import numpy as np

from task2 import DEFAULT_MATRIX, agglomerate


class ClusteringTests(unittest.TestCase):
    def test_assignment_matrices_and_inclusive_threshold(self):
        result = agglomerate(DEFAULT_MATRIX, .85)
        self.assertEqual(result.clusters, ((0, 1, 2, 3), (4,)))
        self.assertEqual([s.similarity for s in result.steps], [.95, .9, .85])
        expected = [
            [[0, .9, .85, .8], [.9, 0, .6, .8], [.85, .6, 0, .7], [.8, .8, .7, 0]],
            [[0, .85, .8], [.85, 0, .7], [.8, .7, 0]],
            [[0, .8], [.8, 0]],
        ]
        for step, matrix in zip(result.steps, expected):
            np.testing.assert_array_equal(step.matrix, matrix)
        self.assertEqual(result.next_similarity, .8)

    def test_high_threshold_does_not_merge(self):
        result = agglomerate(DEFAULT_MATRIX, 1)
        self.assertEqual(len(result.steps), 0)
        self.assertEqual(len(result.clusters), 5)

    def test_zero_threshold_merges_zero_similarities(self):
        result = agglomerate(np.zeros((3, 3)), 0)
        self.assertEqual(result.clusters, ((0, 1, 2),))
        self.assertIsNone(result.next_similarity)
        self.assertEqual(result.steps[0].left, (0,))
        self.assertEqual(result.steps[0].right, (1,))
        np.testing.assert_array_equal(result.steps[-1].matrix, [[0]])

    def test_merge_can_start_outside_first_row(self):
        matrix = np.array([[0, .2, .4], [.2, 0, .9], [.4, .9, 0]])
        result = agglomerate(matrix, .5)
        self.assertEqual(result.clusters, ((0,), (1, 2)))
        np.testing.assert_array_equal(result.steps[0].matrix, [[0, .4], [.4, 0]])

    def test_input_and_snapshots_are_independent(self):
        matrix = DEFAULT_MATRIX.copy()
        result = agglomerate(matrix, 0)
        np.testing.assert_array_equal(matrix, DEFAULT_MATRIX)
        matrix[:] = 0
        np.testing.assert_array_equal(result.initial_matrix, DEFAULT_MATRIX)
        self.assertEqual(result.steps[0].matrix.shape, (4, 4))
        self.assertFalse(np.shares_memory(result.steps[0].matrix, result.steps[1].matrix))

    def test_invalid_inputs(self):
        invalid = [[], [[0]], [[0, .2]], [[0, .2], [.3, 0]],
                   [[1, .2], [.2, 0]], [[0, 1.1], [1.1, 0]],
                   [[0, -.2], [-.2, 0]], [[0, np.nan], [np.nan, 0]],
                   [[0, np.inf], [np.inf, 0]]]
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                agglomerate(matrix)
        for threshold in [-1, 1.1, np.nan, np.inf]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                agglomerate(DEFAULT_MATRIX, threshold)


if __name__ == "__main__":
    unittest.main()
