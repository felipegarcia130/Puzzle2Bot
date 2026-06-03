import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/felipe/puzzlebot_ws/install/obstacle_avoidance_pb'
