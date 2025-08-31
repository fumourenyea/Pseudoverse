import numpy as np

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