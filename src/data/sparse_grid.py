# src/data/sparse_grid.py
class SparseGrid:
    def __init__(self, dimensions, default_value=0.0):
        self.dimensions = dimensions
        self.default_value = default_value
        self.data = {}  # {(x,y): value}
        self.active_regions = set()
        # 简化观察等级
        self.observation_level = [[0]*dimensions[1] for _ in range(dimensions[0])]

    def __getitem__(self, index):
        return self.data.get(tuple(index), self.default_value)

    def __setitem__(self, index, value):
        idx_tuple = tuple(index)
        self.data[idx_tuple] = value
        self.active_regions.add(idx_tuple)
