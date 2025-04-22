"""
A2A (Agent-to-Agent) protocol implementation for SwissKnife.
This module provides a server that exposes SwissKnife agents via the A2A protocol.
"""

from .server import A2AServer
from .registry import AgentRegistry
from .task_manager import MultiAgentTaskManager, AgentTaskManager

__all__ = ["A2AServer", "AgentRegistry", "MultiAgentTaskManager", "AgentTaskManager"]
