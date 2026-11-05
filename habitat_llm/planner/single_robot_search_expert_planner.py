# LLM-free expert for the single robot under partial observability (run with world_model.partial_obs=True): like
# SingleRobotExpertPlanner it takes the goals (which object goes where) from the episode's evaluation propositions, but
# it does not know where the objects are. Before each goal it explores the rooms it has not explored yet, nearest first,
# and stops searching once the goal's object is in its own world graph; only then does it go to the object. Its
# completion time is the reference time t_PO. An object that is never seen (e.g. inside a closed container) fails.
import numpy as np

from habitat_llm.planner.single_robot_expert_planner import SingleRobotExpertPlanner


class SingleRobotSearchExpertPlanner(SingleRobotExpertPlanner):
    def reset(self):
        super().reset()
        self.target = None  # object currently searched for
        self.explored = set()  # rooms already explored in this episode

    def get_plan_dag(self, environment_graph):
        super().get_plan_dag(environment_graph)
        plan, block = [], []
        for a in self.actions_per_agent[0]:
            if isinstance(a, str):  # "Wait <group>" sync points; not needed with one agent
                continue
            block.append(a)
            if a[0] == "Place":
                plan += self.search_block(block)
                block = []
        self.actions_per_agent[0] = plan + self.search_block(block)

    @staticmethod
    def search_block(block):
        """Search for the goal's object, then go to it. Drops the expert's privileged walk to (and opening of) the
        furniture the object is on or in."""
        obj = next((a[1] for a in block if a[0] == "Pick"), None)
        if obj is None:
            return block
        return [("Search", obj, "")] + block[block.index(("Navigate", obj, "")) :]

    def actions_parser(self, next_skill_agents):
        # Before agent 0 starts its next action: turn a Search into Explore actions over the unexplored rooms, skip the
        # remaining Explores once the object is in the robot's graph, and give up a goal whose object was never seen.
        plan, i = self.actions_per_agent[0], self.plan_indx[0] + 1
        if next_skill_agents[0] and not self.is_waiting[0]:
            graph = self.env_interface.world_graph[0]
            while i < len(plan):
                name, arg, _ = plan[i]
                found = self.target is None or graph.has_node(self.target)
                if name == "Search":
                    self.target = arg
                    plan[i : i + 1] = [("Explore", room, "") for room in self.room_tour()]
                elif name == "Explore" and found:
                    i += 1
                elif name != "Explore" and not found:
                    while i < len(plan) and plan[i][0] != "Search":
                        i += 1
                else:
                    break
            if i < len(plan) and plan[i][0] == "Explore":
                self.explored.add(plan[i][1])
            self.plan_indx[0] = i - 1
        return super().actions_parser(next_skill_agents)

    def room_tour(self):
        """Greedy nearest-first order (geodesic) over the unexplored rooms that have furniture, from the robot's
        position. Explore[room] visits the room's furniture in graph order: it enters at the first, leaves at the last."""
        graph = self.env_interface.world_graph[0]
        rooms = {r.name: graph.get_furniture_in_room(r) for r in graph.get_all_rooms() if r.name not in self.explored}
        rooms = {name: furniture for name, furniture in rooms.items() if furniture}
        pos, tour = self.env_interface.sim.agents_mgr[0].articulated_agent.base_pos, []
        while rooms:
            room = min(rooms, key=lambda r: self.geodesic(pos, rooms[r][0].properties["translation"]))
            pos = rooms.pop(room)[-1].properties["translation"]
            tour.append(room)
        return tour

    def geodesic(self, a, b):
        sim = self.env_interface.sim
        a, b = (sim.pathfinder.snap_point(np.array(p), sim._largest_indoor_island_idx) for p in (a, b))
        return sim.geodesic_distance(a, b)
