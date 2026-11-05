#!/usr/bin/env python3

"""
SayTool (TimedHome): lets the planner send a short message to the person who gave the task, e.g. to report a
problem. It changes nothing in the world and takes no simulated time; the message stays in the episode log.
"""

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface

from habitat_llm.tools import PerceptionTool
from habitat_llm.utils.grammar import FREE_TEXT


class SayTool(PerceptionTool):
    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.env_interface = None
        self.skill_config = skill_config

    def set_environment(self, env_interface: "EnvironmentInterface") -> None:
        self.env_interface = env_interface

    @property
    def description(self) -> str:
        return self.skill_config.description

    def process_high_level_action(self, input_query, observations) -> Tuple[None, str]:
        super().process_high_level_action(input_query, observations)
        return None, "Message delivered."

    @property
    def argument_types(self) -> List[str]:
        return [FREE_TEXT]
