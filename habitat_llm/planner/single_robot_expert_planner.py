# LLM-free expert for the single-robot setting: reuses the privileged scripted planner (it reads the episode's
# evaluation propositions and emits Nav/Pick/Nav/Place, opening containers when needed) but gives every action
# to agent 0 so that completion time is that of one robot. Used for t_oracle and as the ceiling anchor.
import os

from habitat_llm.planner.scripted_centralized_planner import ScriptedCentralizedPlanner


class SingleRobotExpertPlanner(ScriptedCentralizedPlanner):
    def __init__(self, plan_config, env_interface):
        super().__init__(plan_config, env_interface)
        # the scripted planner samples among equivalent targets; the seed makes the choice reproducible
        self.seed = int(os.environ.get("TH_EXPERT_SEED", "142"))
        self.reset()

    def get_plan_dag(self, environment_graph):
        super().get_plan_dag(environment_graph)
        merged = []
        for agent_id in sorted(self.actions_per_agent):
            merged += [a for a in self.actions_per_agent[agent_id] if a[0] != "Wait"]
        self.actions_per_agent = {
            agent_id: (merged if agent_id == 0 else []) for agent_id in self.actions_per_agent
        }
