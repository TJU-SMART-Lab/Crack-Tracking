import configparser


class Config(configparser.ConfigParser):
    def __init__(self):
        super().__init__()
        self.read('./config.ini')

