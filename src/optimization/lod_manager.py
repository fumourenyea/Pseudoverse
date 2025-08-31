import numpy as np

class LODManager:
    """动态细节管理器：根据观测等级分配计算资源"""
    
    def __init__(self, config=None):
        self.config = config or {}
        print("初始化LOD管理器")
        
    def update_observation_levels(self, grid, observer_pos):
        """根据观测者位置更新观测级别"""
        x, y = observer_pos
        nx, ny = grid.shape()
        
        # 计算到观测者的距离
        dist_x = np.arange(nx) - x
        dist_y = np.arange(ny) - y
        distance_matrix = np.sqrt(dist_x[:, np.newaxis]**2 + dist_y[np.newaxis, :]**2)
        
        # 基于距离设置观测级别
        grid.observation_level = np.zeros_like(grid.rho, dtype=int)
        grid.observation_level[distance_matrix < 15] = 2  # 直接观测
        grid.observation_level[(distance_matrix >= 15) & (distance_matrix < 40)] = 1  # 仪器观测
        
        return grid