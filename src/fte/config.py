"""This module contains the general configuration of the project."""
from pathlib import Path


SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()
TESTS = ROOT.joinpath("tests").resolve()

__all__ = ["SRC", "ROOT", "TESTS"]
