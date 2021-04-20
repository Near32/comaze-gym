from typing import List, Optional
import math

from astar import find_path
from sys import argv


# Adapted from Kai Liebscher:
class CoMazeObject:
    @classmethod
    def from_dict(cls, data):
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            data = data.__dict__
        return cls(**data)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CoMazeObject) else v for k, v in self.__dict__.items()}


# Adapted from Kai Liebscher:
class Int2D(CoMazeObject):
    def __init__(self, x: int, y: int, **kwargs):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Int2D) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return 31 * hash(self.x) + hash(self.y)


# Adapted from Kai Liebscher:
class Wall(CoMazeObject):
    def __init__(self, position: Int2D, direction: str, **kwargs):
        self.position = Int2D.from_dict(position)
        self.direction = direction

    def __eq__(self, other):
        return isinstance(other, Wall) and self.position == other.position and self.direction == other.direction

    def __hash__(self):
        return 31 * hash(self.position) + hash(self.direction)


# Adapted from Kai Liebscher:
class Goal(CoMazeObject):
    def __init__(self, position: Int2D, color: str, **kwargs):
        self.position = Int2D.from_dict(position)
        self.color = color

    def __eq__(self, other):
        return isinstance(other, Goal) and self.position == other.position and self.color == other.color

    def __hash__(self):
        return 31 * hash(self.position) + hash(self.color)


# Adapted from Kai Liebscher:
class BonusTime(CoMazeObject):
    def __init__(self, position: Int2D, amount: int, **kwargs):
        self.position = Int2D.from_dict(position)
        self.amount = amount

"""
# Adapted from Kai Liebscher:
class GameConfig(CoMazeObject):
    def __init__(self, arenaSize: Int2D, walls: List[Wall], goals: List[Goal], bonusTimes: List[BonusTime],
                 agentStartPosition: Int2D, initialMaxMoves: int, hasSecretGoalRules: bool, **kwargs):
        self.arenaSize = Int2D.from_dict(arenaSize)
        self.walls = {Wall.from_dict(wall) for wall in walls}
        self.goals = [Goal.from_dict(goal) for goal in goals]
        self.bonusTimes = [BonusTime.from_dict(bonusTime) for bonusTime in bonusTimes]
        self.agentStartPositions = Int2D.from_dict(agentStartPosition)
        self.initialMaxMoves = initialMaxMoves
        self.hasSecretGoalRules = hasSecretGoalRules
"""

# Adapted from Kai Liebscher:
class GameState(CoMazeObject):
    def __init__(self, running: bool, won: bool, lost: bool, lostMessage: str, over: bool, started: bool, **kwargs):
        self.running = running
        self.won = won
        self.lost = lost
        self.lostMessage = lostMessage
        self.over = over
        self.started = started


# Adapted from Kai Liebscher:
class SecretGoalRule(CoMazeObject):
    def __init__(self, earlierGoal: Goal, laterGoal: Goal, **kwargs):
        self.earlierGoal = Goal.from_dict(earlierGoal)
        self.laterGoal = Goal.from_dict(laterGoal)


# Adapted from Kai Liebscher:
class Direction:
    def __init__(self, name: str, other_name: str, offset_x: int, offset_y: int):
        self.name = name
        self.other_name = other_name
        self.offset_x = offset_x
        self.offset_y = offset_y

# Adapted from Kai Liebscher:
class Move:
    def __init__(self, action: str, predicted_action: Optional[str] = None, symbol_message: Optional[str] = None,
                 predicted_goal: Optional[str] = None):
        self.action = action
        self.predicted_action = predicted_action
        self.predicted_goal = predicted_goal
        self.symbol_message = symbol_message

"""
# Adapted from Kai Liebscher:
class Player(CoMazeObject):
    def __init__(self, name: str, uuid: str, directions: List[str], lastAction: str, actions: List[str],
                 secretGoalRule: Optional[SecretGoalRule] = None, **kwargs):
        self.name = name
        self.uuid = uuid
        self.directions = directions
        self.lastAction = lastAction
        self.actions = actions
        self.secretGoalRule = SecretGoalRule.from_dict(secretGoalRule) if secretGoalRule else None


# Adapted from Kai Liebscher:
class Game(CoMazeObject):
    def __init__(self, config: GameConfig, state: GameState, name: str, uuid: str, agentPosition: Int2D, usedMoves: int,
                 unreachedGoals: List[Goal], unusedBonusTimes: List[BonusTime], players: List[Player],
                 numOfPlayerSlots: int, mayStillMove: bool, currentPlayer: Player, maxMoves: int, movesLeft: int,
                 **kwargs):
        self.config = GameConfig.from_dict(config)
        self.state = GameState.from_dict(state)
        self.name = name
        self.uuid = uuid
        self.agentPosition = Int2D.from_dict(agentPosition)
        self.usedMoves = usedMoves
        self.unreachedGoals = [Goal.from_dict(goal) for goal in unreachedGoals]
        self.unusedBonusTimes = [BonusTime.from_dict(bonusTime) for bonusTime in unusedBonusTimes]
        self.players = [Player.from_dict(player) for player in players]
        self.numOfPlayerSlots = numOfPlayerSlots
        self.mayStillMove = mayStillMove
        self.currentPlayer = Player.from_dict(currentPlayer)
        self.maxMoves = maxMoves
        self.movesLeft = movesLeft
"""

# Adapted from Kai Liebscher:
class Player(CoMazeObject):
    def __init__(self, directions: List[str], lastAction: str, actions: List[str],
                 secretGoalRule: Optional[SecretGoalRule] = None):
        self.directions = directions
        #self.lastAction = lastAction
        self.actions = actions
        self.secretGoalRule = SecretGoalRule.from_dict(secretGoalRule) if secretGoalRule else None


# Adapted from Kai Liebscher:
class GameConfig(CoMazeObject):
    def __init__(
        self, 
        arenaSize: Int2D, 
        walls: List[Wall], 
        goals: List[Goal], 
        bonusTimes: List[BonusTime],
        agentStartPosition: Int2D, 
        initialMaxMoves: int, 
        hasSecretGoalRules: bool, **kwargs):
        self.arenaSize = Int2D.from_dict(arenaSize)
        
        self.walls = {Wall.from_dict(wall) for wall in walls}
        self.goals = [Goal.from_dict(goal) for goal in goals]
        self.bonusTimes = [BonusTime.from_dict(bonusTime) for bonusTime in bonusTimes]
        self.agentStartPositions = Int2D.from_dict(agentStartPosition)
        self.initialMaxMoves = initialMaxMoves
        self.hasSecretGoalRules = hasSecretGoalRules

# Adapted from Kai Liebscher:
class Game(CoMazeObject):
    def __init__(
        self, 
        config: GameConfig, 
        #state: GameState, 
        agentPosition: Int2D, 
        #usedMoves: int,
        unreachedGoals: List[Goal], 
        unusedBonusTimes: List[BonusTime], 
        #players: List[Player],
        #numOfPlayerSlots: int, 
        #mayStillMove: bool, 
        currentPlayer: Player, 
        #maxMoves: int, 
        #movesLeft: int,
        lastMove: Move,
        **kwargs):
        self.config = GameConfig.from_dict(config)
        #self.state = GameState.from_dict(state)
        self.agentPosition = Int2D.from_dict(agentPosition)
        #self.usedMoves = usedMoves
        self.unreachedGoals = [Goal.from_dict(goal) for goal in unreachedGoals]
        self.unusedBonusTimes = [BonusTime.from_dict(bonusTime) for bonusTime in unusedBonusTimes]
        #self.players = [Player.from_dict(player) for player in players]
        #self.numOfPlayerSlots = numOfPlayerSlots
        #self.mayStillMove = mayStillMove
        self.currentPlayer = Player.from_dict(currentPlayer)
        #self.maxMoves = maxMoves
        #self.movesLeft = movesLeft
        self.lastMove = lastMove
