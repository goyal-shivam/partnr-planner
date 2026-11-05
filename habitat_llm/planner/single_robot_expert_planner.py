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
        # optional forced goal order: info.timedhome.expert_order = [object handle, ...] (first goal first)
        episode = self.env_interface.env.env.env._env.current_episode
        order = (episode.info.get("timedhome") or {}).get("expert_order")
        if order:
            merged = self.reorder_goals(merged, order)
        self.actions_per_agent = {
            agent_id: (merged if agent_id == 0 else []) for agent_id in self.actions_per_agent
        }

    def reorder_goals(self, actions, order):
        """Split the plan into one block per goal (each ends with its Place) and sort the blocks by `order`."""
        names = [self.env_interface.perception.sim_handle_to_name.get(h, h) for h in order]
        blocks, cur = [], []
        for a in actions:
            if isinstance(a, str):  # "Wait <group>" sync points; not needed with one agent
                continue
            cur.append(a)
            if a[0] == "Place":
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)

        def rank(block):
            obj = next((b[1].split(",")[0].strip() for b in block if b[0] == "Place"), None)
            return names.index(obj) if obj in names else len(names)

        return [a for block in sorted(blocks, key=rank) for a in block]
