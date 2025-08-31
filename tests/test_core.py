import unittest
from src.core.engine import CoreEngine
from src.data.sparse_grid import SparseGrid

class TestCoreEngine(unittest.TestCase):
    def setUp(self):
        self.grid = SparseGrid(dimensions=(10,10))
        self.engine = CoreEngine(grid=self.grid)

    def test_step_increments(self):
        initial_step = self.engine.current_step
        self.engine.step()
        self.assertEqual(self.engine.current_step, initial_step + 1)

    def test_run_stops(self):
        self.engine.run(steps=3)
        self.assertEqual(self.engine.current_step, 3)
