"""
AESW Interactive Simulation API
================================
Provides the FastAPI application layer, data models, and simulation session manager
for the Adaptive Evidence-Sharing Walkers research platform.
"""

from aesw.api.app import create_app, app
from aesw.api.simulation_manager import SimulationManager, SimulationSession

__all__ = ["create_app", "app", "SimulationManager", "SimulationSession"]
