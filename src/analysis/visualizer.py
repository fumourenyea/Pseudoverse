import matplotlib.pyplot as plt
import os
import numpy as np

class SimulationVisualizer:
    """模拟可视化器"""
    
    def __init__(self, grid, config=None):
        self.grid = grid
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.fig = None
        print("初始化模拟可视化器")
        
    def update(self, grid, step):
        """更新可视化"""
        if self.fig is None:
            plt.ion()  # 开启交互模式
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.im = self.ax.imshow(grid.rho, cmap='inferno', origin='lower', interpolation='nearest')
            self.fig.colorbar(self.im, ax=self.ax, label='Density')
            self.ax.set_title(f"Step {step}")
            plt.show(block=False)
        else:
            self.im.set_data(grid.rho)
            self.im.set_clim(vmin=np.min(grid.rho), vmax=np.max(grid.rho))
            self.ax.set_title(f"Step {step}")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        
        print(f"更新可视化，步骤: {step}")
        
    def save_final_image(self, grid, step):
        """保存最终图像"""
        output_path = os.path.join(self.output_dir, f"universe_simulation_step_{step}.png")
        
        plt.ioff()  # 关闭交互模式
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid.rho, cmap='inferno', origin='lower', interpolation='nearest')
        fig.colorbar(im, ax=ax, label='Density')
        ax.set_title(f"Universe Simulation - Final State (Step {step})")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return output_path