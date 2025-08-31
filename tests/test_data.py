import unittest
from src.data.sparse_grid import SparseGrid

class TestSparseGrid(unittest.TestCase):
    def setUp(self):
        self.grid = SparseGrid((3,3))

    def test_set_get_item(self):
        self.grid[1,1] = 42
        self.assertEqual(self.grid[1,1], 42)

    def test_default_value(self):
        self.assertEqual(self.grid[0,0], 0.0)
