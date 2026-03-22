"""
Learning subpackage — reinforcement learning, consolidation, and
meta-cognitive update mechanisms.

Brain analog: The synaptic plasticity machinery that modifies connection
strengths based on experience:
- Reward processing  → VTA/substantia nigra dopaminergic RPE signals
- Consolidation      → hippocampal replay during slow-wave sleep
- Skill acquisition  → basal ganglia habit formation
- Experience replay  → hippocampal sharp-wave ripple reactivation

The learning subsystem drives long-term changes in the memory systems;
it reads from episodic memory and writes distilled knowledge into semantic
and procedural stores.
"""

from mnemon.learning.consolidation import ConsolidationEngine
from mnemon.learning.replay import PrioritizedReplayBuffer, ReplayExperience, SumTree
from mnemon.learning.reward import RewardProcessor
from mnemon.learning.skill_acquirer import SkillAcquirer, SkillNeed

__all__ = [
    "ConsolidationEngine",
    "PrioritizedReplayBuffer",
    "ReplayExperience",
    "RewardProcessor",
    "SkillAcquirer",
    "SkillNeed",
    "SumTree",
]
