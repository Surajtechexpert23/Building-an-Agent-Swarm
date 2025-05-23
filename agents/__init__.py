"""Agent imports and initialization module."""
from agents.router import route_message
from agents.knowledge import knowledge_agent
from agents.support import customer_support_agent
from agents.personality import personality_agent

__all__ = [
    'route_message',
    'knowledge_agent',
    'customer_support_agent',
    'personality_agent'
]
