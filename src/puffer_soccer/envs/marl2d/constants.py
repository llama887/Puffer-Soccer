import math

MAX_PLAYERS_PER_TEAM = 11
MAX_PLAYERS = 22
MAX_BALL_SPEED = 5.0
DEFAULT_GAME_LENGTH = 400
DEFAULT_VISION_RANGE = math.pi

FIELD_SIZE = (110.0, 76.0)
IN_FIELD_SIZE = (100.0, 70.0)
GOAL_SIZE = (3.0, 40.0)
POS_NORM = IN_FIELD_SIZE[0] / 2.0

# Goalie, defenders, midfielders, attackers.
INIT_POSITION_11 = [
    (0.0, -0.45),
    (-0.225, -0.3),
    (-0.075, -0.3),
    (0.075, -0.3),
    (0.225, -0.3),
    (-0.2, -0.2),
    (0.0, -0.2),
    (0.2, -0.2),
    (-0.2, -0.1),
    (0.0, -0.1),
    (0.2, -0.1),
]
