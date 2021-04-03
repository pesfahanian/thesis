import os
import json


CONFIG_PATH = f'{os.getcwd()}/config/config.json'

config = json.load(open(CONFIG_PATH))

class Config:
    hidden_channels = config['hidden_channels']
    epochs = config['epochs']
