import numpy as np
import time

class CoreEngine:
    def __init__(self, grid, dt=0.01):
        self.grid = grid
        self.dt = dt
        self.current_step = 0
        self.scale_factor = 1.0
        self.start_time = time.time()
        self.performance_stats = {
            'physics_time': 0,
            'optimization_time': 0,
            'total_time': 0
        }
        
    def step(self, physics_modules, optimization_modules):
        """执行一个模拟步"""
        step_start = time.time()
        
        # 执行物理计算
        physics_start = time.time()
        physics_modules['fields'].evolve(self.grid, self.dt)
        physics_time = time.time() - physics_start
        
        # 应用优化策略
        optimization_start = time.time()
        optimization_modules['lod'].update_observation_levels(self.grid, (self.grid.width//2, self.grid.height//2))
        self.grid.rho, _ = optimization_modules['compressor'].find_and_compress(self.grid.rho)
        self.scale_factor, expanded, _ = optimization_modules['expansion'].check_and_expand(
            self.grid.pressure, self.scale_factor)
        optimization_time = time.time() - optimization_start
        
        # 更新性能统计
        self.performance_stats['physics_time'] += physics_time
        self.performance_stats['optimization_time'] += optimization_time
        self.performance_stats['total_time'] += time.time() - step_start
        
        self.current_step += 1
        
    def get_performance_stats(self):
        """获取性能统计"""
        stats = self.performance_stats.copy()
        if self.current_step > 0:
            stats['avg_physics_time'] = stats['physics_time'] / self.current_step
            stats['avg_optimization_time'] = stats['optimization_time'] / self.current_step
            stats['avg_total_time'] = stats['total_time'] / self.current_step
        return stats