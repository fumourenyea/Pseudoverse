# run_simulation.py
import sys
import os
import yaml
import numpy as np
import time
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# --- 内联定义所有缺失的类，确保代码自包含 ---

class ComputationalGrid:
    """计算网格类，用于存储模拟的各类场数据"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # 初始化各种场
        self.rho = np.random.normal(1.0, 0.1, (width, height))  # 密度场
        self.pressure = np.zeros((width, height))               # 压力场
        self.vx = np.zeros((width, height))                     # x方向速度场
        self.vy = np.zeros((width, height))                     # y方向速度场
        self.phi = np.zeros((width, height))                    # 引力势场
        self.element_abundance = np.zeros((width, height))      # 元素丰度场
        self.observation_level = np.zeros((width, height), dtype=int) # 观测级别
        self.compressed_mask = np.zeros((width, height), dtype=bool)  # 压缩掩码
        
    def shape(self):
        """返回网格的形状"""
        return self.rho.shape
        
    def to_dict(self):
        """将网格数据转换为字典"""
        return {
            'rho': self.rho,
            'pressure': self.pressure,
            'vx': self.vx,
            'vy': self.vy,
            'phi': self.phi,
            'element_abundance': self.element_abundance,
            'observation_level': self.observation_level,
            'compressed_mask': self.compressed_mask
        }
        
    def apply_mask(self):
        """应用压缩掩码，将被压缩的区域清零"""
        self.rho[self.compressed_mask] = 0.0
        self.vx[self.compressed_mask] = 0.0
        self.vy[self.compressed_mask] = 0.0
        self.pressure[self.compressed_mask] = 0.0
        self.element_abundance[self.compressed_mask] = 0.0


class QuantumFieldTheoryEngine:
    """量子场论引擎：模拟基础场的演化"""
    
    def __init__(self, field_names=['electron', 'quark_up', 'quark_down', 'Higgs']):
        self.field_names = field_names
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
        grid.rho = np.maximum(grid.rho, 0.0)  # 确保密度非负
        
        # 更新压力场（简化状态方程）
        grid.pressure = 0.5 * np.power(grid.rho, 1.4)
    
    def _laplacian(self, f):
        """计算二维数组的拉普拉斯算子"""
        return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4 * f)
    
    def _gradient(self, f):
        """计算梯度"""
        gx = (np.roll(f, 1, axis=0) - np.roll(f, -1, axis=0)) / 2.0
        gy = (np.roll(f, 1, axis=1) - np.roll(f, -1, axis=1)) / 2.0
        return gx, gy


class Hadronizer:
    """强子化模块"""
    def __init__(self):
        print("初始化强子化模块")


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


class BlackHoleCompressor:
    """黑洞压缩器：将高密度区域的数据压缩为极简表示"""
    
    def __init__(self, threshold=5.0):
        self.threshold = threshold
        self.black_holes = []
        print(f"初始化黑洞压缩器，阈值: {threshold}")
        
    def find_and_compress(self, density_field):
        """查找高密度区域并将其压缩为黑洞"""
        high_density_coords = np.argwhere(density_field > self.threshold)
        new_black_holes = []
        
        for coord in high_density_coords:
            x, y = coord
            if density_field[x, y] > self.threshold:
                # 计算黑洞质量（原区域的总质量）
                mass = np.sum(density_field[max(0, x-1):min(density_field.shape[0], x+2), 
                                           max(0, y-1):min(density_field.shape[1], y+2)])
                # 创建新黑洞
                new_black_holes.append([mass, x, y])
                # 清除原区域数据
                density_field[max(0, x-1):min(density_field.shape[0], x+2), 
                             max(0, y-1):min(density_field.shape[1], y+2)] = 0.0
                print(f"黑洞形成于 ({x}, {y})，质量: {mass:.2f}")
        
        self.black_holes.extend(new_black_holes)
        return density_field, new_black_holes


class ExpansionController:
    """宇宙膨胀控制器：系统负载过高时触发膨胀以降低密度"""
    
    def __init__(self, trigger_pressure=1.5, expansion_rate=0.01):
        self.trigger_pressure = trigger_pressure
        self.expansion_rate = expansion_rate
        self.expansion_count = 0
        print(f"初始化膨胀控制器，触发压力: {trigger_pressure}, 膨胀率: {expansion_rate}")
        
    def check_and_expand(self, pressure_field, scale_factor):
        """检查平均压力，决定是否触发膨胀"""
        avg_pressure = np.mean(pressure_field)
        
        if avg_pressure > self.trigger_pressure:
            print(f"系统负载过高 (P_avg={avg_pressure:.2f} > {self.trigger_pressure})，触发宇宙膨胀")
            old_scale_factor = scale_factor
            scale_factor *= (1.0 + self.expansion_rate)
            
            # 计算膨胀比例并调整密度 (ρ ∝ a^{-3})
            expansion_ratio = (scale_factor / old_scale_factor) ** 3
            self.expansion_count += 1
            
            print(f"宇宙膨胀: a={old_scale_factor:.3f} -> {scale_factor:.3f}")
            print(f"这是第 {self.expansion_count} 次膨胀事件")
            return scale_factor, True, expansion_ratio
        
        return scale_factor, False, 1.0


class HotfixManager:
    """热修复管理器：处理规则更新与记忆协调"""
    
    def __init__(self):
        self.physical_constants = {'G': 1.0, 'c': 1.0, 'ħ': 0.1}
        self.version = "Physics.v1.0_Standard"
        self.memory_correction_log = []
        print("初始化热修复管理器")
        
    def apply_hotfix(self, fix_name, **kwargs):
        """应用热修复更新物理规则"""
        print(f"应用热修复: {fix_name}")
        return True


class InformationControlSystem:
    """信息管控系统：检测并处理异常实体"""
    
    def __init__(self):
        self.mythology_db = {
            "divine_intervention": [
                "The gods smote the unnatural creature with heavenly fire.",
                "A great beast descended, but was slain by the hero {hero}.",
                "A star fell from heaven, bearing a warning to mankind."
            ]
        }
        self.anomalies_log = []
        print("初始化信息管控系统")
        
    def detect_anomalies(self, grid, anomaly_type="reality_breaker"):
        """检测可能揭示模拟本质的异常实体"""
        print("检测异常实体")
        return []
        
    def neutralize_anomaly(self, grid, coord, myth_template="divine_intervention"):
        """中和异常实体并生成神话解释"""
        print("中和异常实体")
        import random
        myth = random.choice(self.mythology_db.get(myth_template, ["Unknown anomaly neutralized."]))
        return grid, myth


class CoreEngine:
    """核心引擎：驱动模拟运行"""
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
        if expanded:
            self.grid.rho /= _  # _ 是 expansion_ratio
        optimization_time = time.time() - optimization_start
        
        # 更新性能统计
        self.performance_stats['physics_time'] += physics_time
        self.performance_stats['optimization_time'] += optimization_time
        self.performance_stats['total_time'] += time.time() - step_start
        
        self.current_step += 1


class SimulationLoop:
    """主循环控制器"""
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
            plt.colorbar(self.im, ax=self.ax, label='Density')
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
        plt.colorbar(im, ax=ax, label='Density')
        ax.set_title(f"Universe Simulation - Final State (Step {step})")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return output_path


# --- 配置和运行函数 ---

def initialize_configs(config_dir="configs"):
    """初始化配置"""
    configs = {}
    if not os.path.exists(config_dir):
        print(f"警告: 配置目录 '{config_dir}' 不存在，使用默认配置")
        return configs
        
    for filename in os.listdir(config_dir):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            path = os.path.join(config_dir, filename)
            try:
                with open(path, "r", encoding='utf-8') as f:
                    config_name = filename.replace(".yaml", "").replace(".yml", "")
                    configs[config_name] = yaml.safe_load(f)
            except UnicodeDecodeError:
                try:
                    with open(path, "r", encoding='gbk') as f:
                        config_name = filename.replace(".yaml", "").replace(".yml", "")
                        configs[config_name] = yaml.safe_load(f)
                except Exception as e:
                    print(f"加载配置文件 {filename} 时出错: {e}")
            except Exception as e:
                print(f"加载配置文件 {filename} 时出错: {e}")
    return configs


def initialize_physics_modules(configs):
    """初始化物理模块"""
    physics_config = configs.get("physics", {})
    fields = physics_config.get("fields", ['electron', 'quark_up', 'quark_down', 'Higgs'])
    field_engine = QuantumFieldTheoryEngine(fields)
    hadronizer = Hadronizer()
    return {
        'fields': field_engine,
        'hadronizer': hadronizer
    }


def initialize_optimization_modules(configs):
    """初始化优化模块"""
    opt_config = configs.get("optimization", {})
    lod = LODManager(opt_config.get("lod", {}))
    compressor = BlackHoleCompressor(threshold=opt_config.get("blackhole_compressor", {}).get("threshold", 5.0))
    expansion = ExpansionController(
        trigger_pressure=opt_config.get("expansion_controller", {}).get("trigger_pressure", 1.5),
        expansion_rate=opt_config.get("expansion_controller", {}).get("expansion_rate", 0.01)
    )
    hotfix = HotfixManager()
    info_control = InformationControlSystem()
    return {
        'lod': lod,
        'compressor': compressor,
        'expansion': expansion,
        'hotfix': hotfix,
        'info_control': info_control
    }


def main():
    # 1. 加载配置
    configs = initialize_configs("configs")
    print("配置加载完成")

    # 2. 初始化计算网格
    grid_size = configs.get("simulation", {}).get("grid_size", [128, 128])
    grid = ComputationalGrid(*grid_size)
    print(f"初始化网格: {grid_size}")

    # 3. 初始化核心引擎
    dt = configs.get("simulation", {}).get("time_step", 0.01)
    engine = CoreEngine(grid, dt=dt)
    print("核心引擎初始化完成")

    # 4. 初始化模块
    physics_modules = initialize_physics_modules(configs)
    optimization_modules = initialize_optimization_modules(configs)
    print("物理和优化模块初始化完成")

    # 5. 初始化可视化器
    try:
        visualizer = SimulationVisualizer(grid, configs.get("visualization", {}))
        print("可视化器初始化完成")
    except Exception as e:
        print(f"初始化可视化器时出错: {e}")
        visualizer = None

    # 6. 初始化主循环
    try:
        sim_loop = SimulationLoop(engine, physics_modules, optimization_modules, visualizer, configs.get("simulation", {}))
        print("主循环初始化完成")
    except Exception as e:
        print(f"初始化主循环时出错: {e}")
        return

    # 7. 启动模拟
    steps = configs.get("simulation", {}).get("total_steps", 1000)
    print(f"开始模拟，总步数: {steps}")
    try:
        sim_loop.run(steps)
        
        # 模拟结束后保存最终图像
        if visualizer:
            output_path = visualizer.save_final_image(grid, steps)
            print(f"宇宙密度图已保存至: {output_path}")
            
        print("模拟完成")
    except KeyboardInterrupt:
        print("模拟被用户中断")
    except Exception as e:
        print(f"模拟运行时出错: {e}")


if __name__ == "__main__":
    main()