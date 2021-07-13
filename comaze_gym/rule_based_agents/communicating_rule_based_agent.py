from typing import List, Optional
import math

import numpy as np 
from astar import find_path
from sys import argv

from comaze_gym.env.comaze import *

goalColor2idx = {
    "RED":0,
    "GREEN":1,
    "YELLOW":2,
    "BLUE":3,
}

idx2GoalColor = dict(zip(goalColor2idx.values(), goalColor2idx.keys()))

# Adapted from ? and Marie Ossenkopf:
class CommunicatingRuleBasedAgent(object):
    def __init__(self):
        self.directions = [
            Direction('UP', 'DOWN', 0, -1),
            Direction('DOWN', 'UP', 0, 1),
            Direction('LEFT', 'RIGHT', -1, 0),
            Direction('RIGHT', 'LEFT', 1, 0),
        ]

        self.secretgoalStr2id = {
            "RED":0, 
            "YELLOW":1, 
            "BLUE":2, 
            "GREEN":3
        }

        self.reset()

    def reset(self):
        self.last_followed_path = None
        self.toxic_field = None
        self.known_toxic_field = None
        self.suspect_toxic = None
        self.last_goals_unreached = []
        self.reached_goals_str = []
        self.game = None
        self.player = None


    def is_path_to_goal(self, path, goal):
        if goal and path:
            return path[-1] == goal
        return False

    def shortest_paths_to_nearest_goals(self, game: Game, player: Player):
        # calculate the paths to ALL goals and take the shortest
        targets = game.unreachedGoals + game.unusedBonusTimes
        if player.secretGoalRule \
        and player.secretGoalRule.earlierGoal in targets \
        and player.secretGoalRule.laterGoal in targets: # ADDITION: to handle continuing games even when a rule is breached.
                targets.remove(player.secretGoalRule.laterGoal)
                self.toxic_field = player.secretGoalRule.laterGoal
        else:
            self.toxic_field = None


        """
        condition = (
            (
                (
                    len(targets) == 3  and \
                    (player.secretGoalRule.earlierGoal.color=="RED" and player.secretGoalRule.laterGoal.color=="YELLOW")
                ) \
                or (
                    len(targets) == 3  and \
                    (player.secretGoalRule.earlierGoal.color=="YELLOW" and player.secretGoalRule.laterGoal.color=="BLUE")
                ) \
            ) \
            and game.agentPosition.x == 4 and game.agentPosition.y==3)

        condition = (((len(targets) == 3  and (player.secretGoalRule.earlierGoal.color=="RED" and player.secretGoalRule.laterGoal.color=="YELLOW")) or (len(targets) == 3  and (player.secretGoalRule.earlierGoal.color=="YELLOW" and player.secretGoalRule.laterGoal.color=="BLUE"))) and game.agentPosition.x == 4 and game.agentPosition.y==3)
        
        if condition:
            import ipdb; ipdb.set_trace()
        """

        possible_paths: List[List[Int2D]]
        possible_paths = [
            self.shortest_path_to_point(game, goal.position)
            for goal in targets
        ]
        possible_paths.sort(key=lambda path: self.sort_key(game, player, path))
        return possible_paths

    def sort_key(self, game: Game, player: Player, path: List[Int2D]):
        return (
            len(path),
            not self.is_path_to_goal(path, player.secretGoalRule and player.secretGoalRule.earlierGoal.position),
            not self.action_available(game, self.action_name(path[0], path[1])),
        )

    def shortest_path_to_point(self, game: Game, goal: Int2D):
        def neighbors_fn(a):
        	return self.neighbors(game, a)
        def heuristic_cost_estimate_fnct(a,b):
        	return self.direct_dist(a, b)
        def is_goal_reached_fnct(a, b):
            return a.x == b.x and a.y == b.y
        paths = find_path(
            start=game.agentPosition,
            goal=goal,
            neighbors_fnct= neighbors_fn,
            heuristic_cost_estimate_fnct= heuristic_cost_estimate_fnct,
            distance_between_fnct= (lambda a, b: abs(a.x - b.x) + abs(a.y - b.y)),
            is_goal_reached_fnct= is_goal_reached_fnct,
		)
        return list(paths)

    def handle_communication(self, game: Game, player: Player):
        lastMove = game.lastMove
        last_message = lastMove.symbol_message
        self.decoded_secret_goal_rule = self.decode_secret_goal_rule(last_message)
        if len(self.decoded_secret_goal_rule) != 2:
            return 

        earlierGoal_color, laterGoal_color = self.decoded_secret_goal_rule
        if hasattr(self, "secretgoalids") and self.secretgoalids is not None:
            self.secretgoalids.append(self.secretgoalStr2id[earlierGoal_color])
            self.secretgoalids.append(self.secretgoalStr2id[laterGoal_color])

        unreachedGoals_colors = [g.color for g in game.unreachedGoals]
        if earlierGoal_color in unreachedGoals_colors:
            # then we need to put later goal as a toxic tile:
            laterGoals = [g for g in game.unreachedGoals if g.color == laterGoal_color]
            if len(laterGoals) == 1:
                self.known_toxic_field = laterGoals[0].position
        else:
            self.known_toxic_field = None 
        
        return

    def next_move(self, game: Game, player: Player):
        self.game = game 
        self.player = player
        # Bookkeeping of the secretGoalRule:
        if player.secretGoalRule is not None:
            self.secretgoalids = [
                self.secretgoalStr2id[player.secretGoalRule.earlierGoal.color],
                self.secretgoalStr2id[player.secretGoalRule.laterGoal.color],
            ]
        else:
            self.secretgoalids = None

        # Decode the message, assuming similar language.
        # Set the toxic tile directly from message.
        self.handle_communication(game, player)
        secret_goal_rule_message = self.encode_secret_goal_rule(player)
        
        paths_to_goals = self.shortest_paths_to_nearest_goals(game, player)

        if not paths_to_goals:
            null_move = Move(
                action="SKIP", 
                predicted_action=None, 
                predicted_goal=None, 
                symbol_message=secret_goal_rule_message,
            )
            return null_move

        # Check if we finished reached goals
        if self.last_goals_unreached != game.unreachedGoals:
            self.suspect_toxic = None
            newly_reached_goals = [goal for goal in self.last_goals_unreached if goal not in game.unreachedGoals]
            if len(newly_reached_goals)==1:    
                self.reached_goals_str.append(newly_reached_goals[0].color)
        self.last_goals_unreached = game.unreachedGoals

        path_to_goal = paths_to_goals[0]  # default
        for path in paths_to_goals:
            # Don't go to suspect toxic path
            """
            if self.is_path_to_goal(path, self.suspect_toxic):
                continue
            """
            if self.known_toxic_field is not None \
            and self.is_path_to_goal(path, self.known_toxic_field):
                continue
            # Set new toxic suspect if we visited the same field twice
            """
            if path == self.last_followed_path:
                self.suspect_toxic = path[-1]
                continue
            """
            path_to_goal = path
            break

        self.last_followed_path = path_to_goal

        self.predicted_goal = self.get_color_name(game, path_to_goal[-1])

        action = self.action_name(path_to_goal[0], path_to_goal[1])
        if self.action_available(game, action):  # we can do this step
            my_action = action
            if len(path_to_goal) <= 2:  # only one step needed to reach a goal
                # FIXME we should inspect the next goal here, but this is a rare case
                # there are initially 4*4+2*3 = 22 / 49 tiles next to goals where this case can hit,
                # half that on average after collecting some goals,
                # half the time we can't do the current step and land in the bottom-most else case,
                # and again half the time the other actually needs to SKIP so the prediction is right.
                # So we can gain about 5-6% accuracy by implementing this.
                prediction = 'SKIP'
            else:
                next_action = self.action_name(path_to_goal[1], path_to_goal[2])
                if self.action_available(game, next_action):  # we have to do the next step, the other needs to SKIP
                    prediction = 'SKIP'
                else:  # other needs to do the next step
                    prediction = next_action
        else:  # the other needs to do this step
            my_action = 'SKIP'
            prediction = action

        move = Move(
            action=my_action, 
            predicted_action=prediction, 
            predicted_goal=self.predicted_goal,
            symbol_message=secret_goal_rule_message,
        )

        return move 

    # -------------Helper Functions--------------

    def get_hidden_state(self):
        """
        returns a hidden state consisting of:
        (i)     three one-hot-encodings of the reached goals.
        (ii)    one-hot-encoding of the predicted goal.
        (iii)   four one-hot-encodings for both players' secret goal rules.
        """
        hs = np.zeros((3+1+4)*4)
        startidx=0

        if self.game is not None:
            unreachedGoals_str = [goal.color for goal in self.game.unreachedGoals]
            """
            reached_goals_str = [
                goal_str 
                for goal_str in self.secretgoalStr2id.keys() 
                if goal_str not in unreachedGoals_str
            ]
            """
            reached_goals_str = self.reached_goals_str
            for goal_str in reached_goals_str:
                hs[ startidx+self.secretgoalStr2id[goal_str] ] = 1.0
                startidx += 4
        startidx = 3*4 
        
        if hasattr(self, "predicted_goal") and self.predicted_goal is not None:
            hs[ startidx+self.secretgoalStr2id[self.predicted_goal] ] = 1.0
        
        startidx += 4
        if hasattr(self, "secretgoalids"):
            for sgid in range(len(self.secretgoalids)):
                hs[startidx+sgid*4+self.secretgoalids[sgid]] = 1.0
        
        return hs

    def encode_secret_goal_rule(self, player: Player):
        secret_goal_rule = player.secretGoalRule
        secret_goal_rule_idx = [
            goalColor2idx[secret_goal_rule.earlierGoal.color],
            goalColor2idx[secret_goal_rule.laterGoal.color],
        ]
        self.secret_goal_rule_message = 4*secret_goal_rule_idx[0]+secret_goal_rule_idx[1]
        return self.secret_goal_rule_message

    def decode_secret_goal_rule(self, secret_goal_rule_message: int):
        if secret_goal_rule_message is None:
            return []

        earlierGoal_idx = secret_goal_rule_message//4
        laterGoal_idx = secret_goal_rule_message%4
        earlierGoal_color = idx2GoalColor[earlierGoal_idx]
        laterGoal_color = idx2GoalColor[laterGoal_idx]
        self.other_secret_goal_rule_colors = [earlierGoal_color, laterGoal_color]
        return self.other_secret_goal_rule_colors

    def get_color_name(self, game, position):
        return next((goal.color for goal in game.unreachedGoals if goal.position == position), None)

    def neighbors(self, game: Game, point: Int2D):
        neighbors = []

        for direction in self.directions:
            if Wall(point, direction.name) in game.config.walls:
                # wall at current position
                continue

            new_x = point.x + direction.offset_x
            new_y = point.y + direction.offset_y
            if new_x < 0 or new_y < 0 or new_x >= game.config.arenaSize.x or new_y >= game.config.arenaSize.y:
                # new point out of bounds
                continue

            new_point = Int2D(new_x, new_y)
            if Wall(new_point, direction.other_name) in game.config.walls:
                # wall from new position
                continue

            if self.toxic_field is not None:
                # Don't calculate path that go over our toxic field
                if new_x == self.toxic_field.position.x \
                and new_y == self.toxic_field.position.y:
                    continue
            """
            if self.known_toxic_field is not None:
                # Don't calculate path that go over our toxic field
                if new_x == self.known_toxic_field.position.x \
                and new_y == self.known_toxic_field.position.y:
                    continue
            """
            neighbors.append(new_point)

        return neighbors

    def action_name(self, start: Int2D, goal: Int2D):
        offset_x = goal.x - start.x
        offset_y = goal.y - start.y
        for direction in self.directions:
            if direction.offset_x == offset_x and direction.offset_y == offset_y:
                return direction.name
        return 'SKIP'

    # Tests if current player can do this action
    def action_available(self, game: Game, test_action_name: str):
        return test_action_name in game.currentPlayer.directions

    # Direct Distance between 2 Points
    def direct_dist(self, start: Int2D, goal: Int2D):
        return math.sqrt((goal.x - start.x) ** 2 + (goal.y - start.y) ** 2)


from ..utils.agent_wrappers import RuleBasedAgentWrapper

def build_WrappedCommunicatingRuleBasedAgent(player_idx:int, action_space_dim:object):
	agent = CommunicatingRuleBasedAgent()
	wrapped_agent = RuleBasedAgentWrapper(
		ruleBasedAgent=agent, 
		player_idx=player_idx, 
		action_space_dim=action_space_dim,
		nbr_actors = 1
	)
	return wrapped_agent