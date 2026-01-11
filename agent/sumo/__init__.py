from .sumo_agent import (
    SumoAgent,
    ManualAgent,
    AggressiveAgent,
    DefensiveAgent,
    RandomAgent,
)

# Backwards compatibility alias
SumoRobot = AggressiveAgent

__all__ = [
    "SumoAgent",
    "SumoRobot",
    "ManualAgent",
    "AggressiveAgent",
    "DefensiveAgent",
    "RandomAgent",
]
