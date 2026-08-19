from enum import Enum, auto

class Gait(Enum):
    TROT = 0
    CRAWL = 1

class Mode(Enum):
    DEACTIVATED = 0
    IDLE = 1
    CALIBRATION = 2
    REST = 3
    STAND = 4
    WALK = 5
