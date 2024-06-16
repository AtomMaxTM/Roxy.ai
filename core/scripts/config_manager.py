import os
from configparser import ConfigParser

def get_config_path():
    current_dir = os.getcwd()

    config_path = os.path.join(current_dir, 'config.ini')

    if os.path.exists(config_path):
        return config_path
    else:
        while not os.path.exists(config_path):
            parent_dir = os.path.dirname(current_dir)

            if parent_dir == current_dir:
                break
            current_dir = parent_dir
            config_path = os.path.join(current_dir, 'config.ini')
            if os.path.exists(config_path):
                break
    return config_path


def get_config():
    config = ConfigParser()
    config.read(get_config_path(), encoding='utf-8')
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    return config_dict

def update_config(upd):
    config = ConfigParser()
    for section, options in upd.items():
        config[section] = options
    with open(get_config_path(), 'w', encoding='utf-8') as configfile:
        config.write(configfile)