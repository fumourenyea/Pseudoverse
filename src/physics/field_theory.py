import numpy as np

class QuantumFieldTheoryEngine:
    """量子场论引擎：模拟基础场的演化"""
    
    def __init__(self, field_names=['electron', 'quark_up', 'quark_down', 'Higgs']):
        self.field_names = field_names
        self.fields = {name: np.zeros((100, 100)) for name in field_names}
        print(f"初始化量子场论引擎，包含场: {field_names}")
        
    def evolve(self, grid, dt):
        """演化场：包括扩散和引力作用"""
        # 扩散过程
        laplacian = self._laplacian(grid.rho)
        diffusion = 0.1 * laplacian
        
        # 简单的引力效应模拟
        gx, gy = self._gradient(grid.rho)
        gravity_effect = 0.05 * (gx + gy)
        
        # 更新密度场
        grid.rho += (diffusion + gravity_effect) * dt
        
        # 确保密度非负
        grid.rho = np.maximum(grid.rho, 0.0)
        
        # 更新压力场（简化状态方程）
        grid.pressure = 0.5 * np.power(grid.rho, 1.4)
    
    def _laplacian(self, f):
        """计算二维数组的拉普拉斯算子"""
        return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4 * f)
    
    def _gradient(self, f):
        """计算梯度"""
        gx = np.roll(f, 1, axis=0) - np.roll(f, -1, axis=0)
        gy = np.roll(f, 1, axis=1) - np.roll(f, -1, axis=1)
        return gx, gy