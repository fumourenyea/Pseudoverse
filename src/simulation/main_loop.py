import time

class SimulationLoop:
    def __init__(self, engine, physics_modules, optimization_modules, visualizer, config):
        self.engine = engine
        self.physics_modules = physics_modules
        self.optimization_modules = optimization_modules
        self.visualizer = visualizer
        self.config = config
        
    def run(self, steps):
        """运行主模拟循环"""
        visualization_interval = self.config.get("visualization_interval", 20)
        
        for step in range(steps):
            # 执行物理计算和优化策略
            self.engine.step(self.physics_modules, self.optimization_modules)
            
            # 更新可视化
            if self.visualizer and step % visualization_interval == 0:
                self.visualizer.update(self.engine.grid, step)
                
            # 添加一个小延迟，使模拟不会运行得太快
            time.sleep(0.001)