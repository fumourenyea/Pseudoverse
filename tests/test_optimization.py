import unittest
from src.optimization.lod_manager import LODManager
from src.data.sparse_grid import SparseGrid

class TestLODManager(unittest.TestCase):
    def setUp(self):
        self.grid = SparseGrid((5,5))
        self.lod = LODManager()

    def test_allocate_resources_no_error(self):
        # 默认grid没有 observation_level 属性时也不报错
        try:
            self.lod.allocate_resources(self.grid)
        except Exception as e:
            self.fail(f"allocate_resources raised {type(e)} unexpectedly!")
