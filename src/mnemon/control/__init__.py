"""
Control subpackage — cognitive orchestration, attention gating, goal management,
and the central executive loop.

Brain analog: The prefrontal executive network that integrates signals from all
other brain regions and drives goal-directed behaviour:
- Orchestrator        → lateral prefrontal cortex (cognitive cycle execution)
- Attention control   → basal forebrain cholinergic system (selective gating)
- Goal manager        → anterior prefrontal cortex (hierarchical goal tracking)
- Meta-cognition      → anterior cingulate cortex (error monitoring and strategy)

All control modules are stateless with respect to each other; shared state
passes through working memory and the CognitiveBus exclusively.
"""
