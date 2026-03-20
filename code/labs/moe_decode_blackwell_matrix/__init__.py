"""Matrix/playbook lab for Blackwell MoE decode tradeoff studies."""

from .matrix_catalog import available_playbooks, load_playbook
from .matrix_types import MatrixPlaybook, MatrixScenario

__all__ = [
    "MatrixPlaybook",
    "MatrixScenario",
    "available_playbooks",
    "load_playbook",
]
