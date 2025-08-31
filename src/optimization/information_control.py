import numpy as np
import random

class InformationControlSystem:
    """信息管控系统：检测并处理异常实体"""
    
    def __init__(self):
        self.mythology_db = {
            "divine_intervention": [
                "The gods smote the unnatural creature with heavenly fire.",
                "A great beast descended, but was slain by the hero {hero}.",
                "A star fell from heaven, bearing a warning to mankind."
            ],
            "chinese_mythology": [
                "天神降怒，以天火净化不洁之物",
                "巨兽{name}现世，吞食天地，后为英雄{hero}所斩",
                "天降陨石，上书箴言「{warning}」"
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
        myth = random.choice(self.mythology_db.get(myth_template, ["Unknown anomaly neutralized."]))
        return grid, myth