"""The built-in general-purpose agent fleet.

Each agent is bound to an open-weight model chosen for its strengths:
    planner, summarizer   -> kimi   (long-context reasoning / synthesis)
    researcher, coder     -> qwen   (general + coding)
    reviewer, critic      -> llama  (balanced judgement)
Re-bind any of these in config without changing code.
"""

from agentos.agents.builtin.coder import CoderAgent
from agentos.agents.builtin.critic import CriticAgent
from agentos.agents.builtin.planner import PlannerAgent
from agentos.agents.builtin.researcher import ResearcherAgent
from agentos.agents.builtin.reviewer import ReviewerAgent
from agentos.agents.builtin.summarizer import SummarizerAgent

ALL_AGENTS = [
    PlannerAgent,
    ResearcherAgent,
    CoderAgent,
    ReviewerAgent,
    CriticAgent,
    SummarizerAgent,
]

__all__ = [
    "PlannerAgent",
    "ResearcherAgent",
    "CoderAgent",
    "ReviewerAgent",
    "CriticAgent",
    "SummarizerAgent",
    "ALL_AGENTS",
]
