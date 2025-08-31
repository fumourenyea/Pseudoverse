import numpy as np

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