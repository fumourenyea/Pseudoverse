import numpy as np

class ComputationalGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.rho = np.zeros((width, height))  # 密度场
        self.pressure = np.zeros((width, height))  # 压力场
        self.observation_level = np.zeros((width, height), dtype=int)  # 观测级别