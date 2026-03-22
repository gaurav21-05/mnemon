"""
Memory subpackage — concrete implementations of the long-term and working
memory subsystems defined in ``mnemon.core.interfaces``.

Brain analog: The distributed memory system of the mammalian brain:
- Episodic memory  → hippocampal formation (context-dependent recall)
- Semantic memory  → neocortical association areas (facts and concepts)
- Procedural memory → basal ganglia / cerebellum (skills and habits)
- Working memory   → dorsolateral prefrontal cortex (active context)
- Valence memory   → amygdala (emotional salience tagging)

Each subsystem is independently replaceable: they share no state directly
and communicate exclusively through the CognitiveBus.
"""
