from typing import List, Optional
import copy
import numpy as np 

from comaze_gym.env.comaze import CoMazeObject, Int2D, Wall, Goal
from comaze_gym.env.comaze import SecretGoalRule, BonusTime, Player, GameConfig
from comaze_gym.env.comaze import Game

actions_idx2str = {
    0:"LEFT",
    1:"RIGHT",
    2:"UP",
    3:"DOWN",
    4:"SKIP",
}

class RuleBasedAgentWrapper(object):
    def __init__(self, ruleBasedAgent:object, player_idx:int, action_space_dim:object, nbr_actors:int):
        self.nbr_actors = nbr_actors
        self.action_space_dim = action_space_dim
        self.vocab_size = (self.action_space_dim-1)//5
        
        self.actions_idx2str = {
            0*self.vocab_size:"LEFT",
            1*self.vocab_size:"RIGHT",
            2*self.vocab_size:"UP",
            3*self.vocab_size:"DOWN",
            4*self.vocab_size:"SKIP",
        }

        self.actions_str2idx = dict(zip(self.actions_idx2str.values(), self.actions_idx2str.keys()))

        self.training = False
        self.player_idx = player_idx
        self.original_ruleBasedAgent = ruleBasedAgent
        self.ruleBasedAgents = []
        self.reset_actors()

    def clone(self, **kwargs):
        cloned_agent = copy.deepcopy(self)
        cloned_agent.reset_actors()
        return cloned_agent

    @property
    def handled_experiences(self):
        return 0

    @handled_experiences.setter
    def handled_experiences(self, val):
        pass

    def get_experience_count(self):
        return self.handled_experiences

    def get_update_count(self):
        return 0

    def set_nbr_actor(self, nbr_actors:int):
        self.nbr_actors = nbr_actors
        self.reset_actors()

    def reset_actors(self, indices:List[int]=None):
        if indices is None: indices = list(range(self.nbr_actors))
        
        for idx in indices:
            if len(self.ruleBasedAgents) <= idx:
                self.ruleBasedAgents.append(copy.deepcopy(self.original_ruleBasedAgent))
                continue
            self.ruleBasedAgents[idx] = copy.deepcopy(self.original_ruleBasedAgent)
        
    def take_action(self, state, infos):
        """
        Convert the :param state: and :param infos:
        into the input that the rule-based agent expects. 
        """

        ### TODO: batch of obs
        actions = np.asarray([
            self.action_space_dim-1 for _ in range(self.nbr_actors)
            ]
        )
        
        for pidx in range(self.nbr_actors):
            if self.player_idx != infos[pidx]["abstract_repr"]["current_player"]: continue
            
            goals = {
                Goal(position=Int2D(*goal_pos), color=goal_str) 
                for goal_str, goal_pos in infos[pidx]["abstract_repr"]["goals"].items()
            }
            config = GameConfig(
                arenaSize=infos[pidx]["abstract_repr"]["arenaSize"],
                walls=[],
                goals=goals,
                bonusTimes=[],
                agentStartPosition=Int2D(0,0),
                initialMaxMoves=100,
                hasSecretGoalRules=len(infos[pidx]["abstract_repr"]["secretGoalRule"])>0,
            )
            agentPosition = infos[pidx]["abstract_repr"]["agentPosition"]
            reachedGoals = infos[pidx]["abstract_repr"]["reached_goals"]
            unreachedGoals = [goal for goal in goals if goal.color not in reachedGoals]
            actions_indices = infos[pidx]["abstract_repr"]["actions"][self.player_idx]
            actions_str = [actions_idx2str[idx] for idx in actions_indices]
            
            lastMove = infos[pidx]["abstract_repr"]["last_move"]

            secret_goal_rule = None 
            if config.hasSecretGoalRules:
                secret_goal_rule = infos[pidx]["abstract_repr"]["secretGoalRule"][self.player_idx]

            currentPlayer = Player(
                #directions=infos[pidx]["abstract_repr"]["directions"], 
                # WARNING: these must be Str, not Directions...
                # cf. method action_available in actionOnlyRuleBasedAgents...
                #directions=[ d.name for d in infos[pidx]["abstract_repr"]["directions"]], 
                # WARNING: these must be availableActions actually: 
                # cf. method action_available in actionOnlyRuleBasedAgents...
                directions = actions_str,
                lastAction="SKIP",
                actions=actions_str,
                secretGoalRule=secret_goal_rule,
            )
            game = Game(
                config=config,
                agentPosition=Int2D(*agentPosition),
                unreachedGoals=unreachedGoals,
                unusedBonusTimes=[],
                currentPlayer=currentPlayer,
                lastMove=lastMove,
            )
            player = currentPlayer
            next_move = self.ruleBasedAgents[pidx].next_move(game=game, player=player)
            next_move_str = next_move.action
            
            """
            if next_move_str == "SKIP":
                import ipdb; ipdb.set_trace()
            """
            
            next_symbol_message_idx = next_move.symbol_message
            actions[pidx] = self.actions_str2idx[next_move_str]
            
            if next_symbol_message_idx is not None:
                actions[pidx] += next_symbol_message_idx

        return actions


