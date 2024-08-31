import os
import yaml

class Config:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            if not config:
                raise ValueError("Config file is empty")
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return None

    def get(self, key, default=None):
        return self.config.get(key, default)

config = Config()
