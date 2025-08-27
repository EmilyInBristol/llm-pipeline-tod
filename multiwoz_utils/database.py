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
        """Load database and convert field names to lowercase, extract all field keys."""
        for domain in self.DOMAINS:
            file_path = os.path.join(self.database_path, f"{domain}_db.json")
            with open(file_path, "r", encoding="utf-8") as f:
                self.database_data[domain] = json.load(f)

            self.database_keys[domain] = set()

            if domain == 'taxi':
                # taxi is a dict
                self.database_data[domain] = {
                    k.lower(): v for k, v in self.database_data[domain].items()
                }
                self.database_keys[domain].update(self.database_data[domain].keys())
            else:
                for i, item in enumerate(self.database_data[domain]):
                    # Convert all fields to lowercase
                    self.database_data[domain][i] = {
                        k.lower(): v for k, v in item.items()
                    }
                    self.database_keys[domain].update(self.database_data[domain][i].keys())

    def query(self, domain: Text, constraints: Dict[Text, Text]) -> List[Dict]:
        """
        Return a list of entities in the specified domain that satisfy all constraints.

        Args:
            domain:      Domain name to query, such as 'hotel', 'restaurant'.
            constraints: Hard constraints in key-value pairs, such as {'area': 'north', 'parking': 'yes'}

        Returns:
            List of entities (dictionaries) that satisfy the conditions
        """
        results = []

        if domain not in self.database_data:
            print(f"[Warning] Domain '{domain}' not in database.")
            return results

        entities = self.database_data[domain]

        if domain == "taxi":
            results = [entities]  # Return all directly
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


# ✅ Default database path (can be changed to your real path)
DEFAULT_DATABASE_PATH = "./multiwoz_database"
# ✅ Automatically initialize global database instance (recommended approach)
default_database = MultiWOZDatabase(DEFAULT_DATABASE_PATH)
