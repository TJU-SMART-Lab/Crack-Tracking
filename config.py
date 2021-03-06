import configparser

config = configparser.ConfigParser()

config.read('./config.ini')

img_path = ''
crack_direction = ''
img_time = ''
cycle = ''

alarm_time = ''
alarm_min_distance = ''
alarm_max_distance = ''

# todo 全局config怎么用？
