# multiwoz_utils/database.py

import os
import json
from typing import Dict, Text, List


class MultiWOZDatabase:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.DOMAINS = [
            'restaurant',
            'hotel',
            'attraction',
            'train',
            'taxi',
            'police',
            'hospital'
        ]
        self.database_data = {}
        self.database_keys = {}
        self._load_data()

    def _load_data(self):
        """加载数据库，并将字段名转为小写，提取所有字段 key。"""
        for domain in self.DOMAINS:
            file_path = os.path.join(self.database_path, f"{domain}_db.json")
            with open(file_path, "r", encoding="utf-8") as f:
                self.database_data[domain] = json.load(f)

            self.database_keys[domain] = set()

            if domain == 'taxi':
                # taxi 是 dict
                self.database_data[domain] = {
                    k.lower(): v for k, v in self.database_data[domain].items()
                }
                self.database_keys[domain].update(self.database_data[domain].keys())
            else:
                for i, item in enumerate(self.database_data[domain]):
                    # 所有字段转为小写
                    self.database_data[domain][i] = {
                        k.lower(): v for k, v in item.items()
                    }
                    self.database_keys[domain].update(self.database_data[domain][i].keys())

    def query(self, domain: Text, constraints: Dict[Text, Text]) -> List[Dict]:
        """
        返回指定 domain 中满足所有 constraints 的实体列表。

        参数：
            domain:      查询的领域名，如 'hotel'、'restaurant'。
            constraints: 键值对形式的硬约束，如 {'area': 'north', 'parking': 'yes'}

        返回：
            满足条件的实体（字典）组成的列表
        """
        results = []

        if domain not in self.database_data:
            print(f"[Warning] Domain '{domain}' not in database.")
            return results

        entities = self.database_data[domain]

        if domain == "taxi":
            results = [entities]  # 直接返回所有
        else:
            for entity in entities:
                match = True
                for key, value in constraints.items():
                    key = key.lower()
                    entity_value = entity.get(key, "").lower()
                    if isinstance(entity_value, list):
                        if value.lower() not in [v.lower() for v in entity_value]:
                            match = False
                            break
                    else:
                        if value.lower() != entity_value:
                            match = False
                            break
                if match:
                    results.append(entity)

        return results


# ✅ 默认数据库路径（可以改为你的真实路径）
DEFAULT_DATABASE_PATH = "./multiwoz_database"
# ✅ 自动初始化全局数据库实例（推荐做法）
default_database = MultiWOZDatabase(DEFAULT_DATABASE_PATH)
