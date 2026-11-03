# LLM-free expert for the HUMAN embodiment: like SingleRobotExpertPlanner but every action goes to agent 1 (the
# humanoid), so the completion time is what an ideal human-embodied player could achieve (walking at humanoid speed,
# oracle pick/place). Used to set deadlines for the HITL study; a human at the keyboard is slower than this.
import os

from habitat_llm.planner.scripted_centralized_planner import ScriptedCentralizedPlanner


class SingleHumanExpertPlanner(ScriptedCentralizedPlanner):
    HUMAN_ID = 1

    def __init__(self, plan_config, env_interface):
        super().__init__(plan_config, env_interface)
        self.seed = int(os.environ.get("TH_EXPERT_SEED", "142"))
        self.reset()

    def get_plan_dag(self, environment_graph):
        super().get_plan_dag(environment_graph)
        merged = []
        for agent_id in sorted(self.actions_per_agent):
            merged += [a for a in self.actions_per_agent[agent_id] if a[0] != "Wait"]
        self.actions_per_agent = {
            agent_id: (merged if agent_id == self.HUMAN_ID else []) for agent_id in self.actions_per_agent
        }
