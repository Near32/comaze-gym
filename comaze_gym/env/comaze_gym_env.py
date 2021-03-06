import gym_minigrid
from gym_minigrid.minigrid import *
from comaze_gym.env import CommunicationChannel
from gym import spaces 
import copy
import numpy as np
import cv2
import random

DEBUG = True

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'white' : np.array([255, 255, 255]),
    'black'  : np.array([20, 20, 20]),
}
gym_minigrid.minigrid.COLORS = COLORS

COLOR_NAMES = sorted(list(COLORS.keys()))
gym_minigrid.minigrid.COLOR_NAMES = COLOR_NAMES

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'white' : 6,
    'black' : 7,
}
gym_minigrid.minigrid.COLOR_TO_IDX = COLOR_TO_IDX

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
gym_minigrid.minigrid.IDX_TO_COLOR = IDX_TO_COLOR

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
    'wall_up'       : 11,
    'wall_down'     : 12,
    'wall_left'     : 13,
    'wall_right'    : 14,
    'red_goal'      : 15,
    'yellow_goal'   : 16,
    'blue_goal'     : 18,
    'green_goal'    : 19,
    'time_bonus'    : 20,
    'wall_left_up'  : 21,
    'wall_left_down': 22,
    'wall_right_up' : 23,
    'wall_right_down': 24,
    
}
gym_minigrid.minigrid.OBJECT_TO_IDX = OBJECT_TO_IDX

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
gym_minigrid.minigrid.IDX_TO_OBJECT = IDX_TO_OBJECT


class CoMazeWorldObj(WorldObj):
    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

class CoMazeWall(CoMazeWorldObj):
    def can_go_through(self, dir):
        raise NotImplementedError

    def can_enter_from(self, dir):
        raise NotImplementedError

class WallUp(CoMazeWall):
    def __init__(self, color='grey'):
        super().__init__('wall_up', color)

    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def can_go_through(self, direction):
        if direction != 3:
            return True
        return False

    def can_enter_from(self, direction):
        if direction != 1: #looking down
            return True
        return False
    
    def render(self, img):
        """
        x-axis is from left to right.
        y-axis is from top to bottom.
        """
        #fill_coords(img, point_in_rect(0.25, 0.75, 0, 0.25), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 1, 0, 0.25), COLORS[self.color])
        #fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class WallDown(CoMazeWall):
    def __init__(self, color='grey'):
        super().__init__('wall_down', color)

    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def can_go_through(self, direction):
        if direction != 1:
            return True
        return False
    
    def can_enter_from(self, direction):
        if direction != 3: #looking up
            return True
        return False
    
    def render(self, img):
        """
        x-axis is from left to right.
        y-axis is from top to bottom.
        """
        #fill_coords(img, point_in_rect(0.25, 0.75, 0.75, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 1, 0.75, 1), COLORS[self.color])
        #fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class WallLeft(CoMazeWall):
    def __init__(self, color='grey'):
        super().__init__('wall_left', color)

    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def can_go_through(self, direction):
        if direction != 2:
            return True
        return False

    def can_enter_from(self, direction):
        if direction != 0: #looking right
            return True
        return False
    
    def render(self, img):
        """
        x-axis is from left to right.
        y-axis is from top to bottom.
        """
        #fill_coords(img, point_in_rect(0, 0.25, 0.25, 0.75), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 0.25, 0, 1), COLORS[self.color])

class WallLeftUp(CoMazeWall):
    def __init__(self, color='grey'):
        super().__init__('wall_left_up', color)
    
    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def can_go_through(self, direction):
        if direction != 2 and direction != 3:
            return True
        return False
    
    def can_enter_from(self, direction):
        if direction != 0 and direction != 1: #looking right or down
            return True
        return False
    
    def render(self, img):
        """
        x-axis is from left to right.
        y-axis is from top to bottom.
        """
        #fill_coords(img, point_in_rect(0, 0.25, 0.25, 0.75), COLORS[self.color])
        #fill_coords(img, point_in_rect(0.25, 0.75, 0, 0.25), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 0.25, 0, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 1, 0, 0.25), COLORS[self.color])

class WallLeftDown(CoMazeWall):
    def __init__(self, color='grey'):
        super().__init__('wall_left_down', color)
    
    def can_overlap(self):
        return True

    def see_behind(self):
        return True
 
    def can_go_through(self, direction):
        if direction != 2 and direction != 1:
            return True
        return False
    
    def can_enter_from(self, direction):
        if direction != 0 and direction != 3: #looking right or up
            return True
        return False
    
    def render(self, img):
        """
        x-axis is from left to right.
        y-axis is from top to bottom.
        """
        #fill_coords(img, point_in_rect(0, 0.25, 0.25, 0.75), COLORS[self.color])
        #fill_coords(img, point_in_rect(0.25, 0.75, 0.75, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 0.25, 0, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 1, 0.75, 1), COLORS[self.color])

class WallRight(CoMazeWall):
    def __init__(self, color='grey'):
        super().__init__('wall_right', color)

    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def can_go_through(self, direction):
        if direction != 0:
            return True
        return False
    
    def can_enter_from(self, direction):
        if direction != 2: #looking left
            return True
        return False
    
    def render(self, img):
        """
        x-axis is from left to right.
        y-axis is from top to bottom.
        """
        #fill_coords(img, point_in_rect(0.75, 1, 0.25, 0.75), COLORS[self.color])
        fill_coords(img, point_in_rect(0.75, 1, 0, 1), COLORS[self.color])
        #fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class WallRightUp(CoMazeWall):
    def __init__(self, color='grey'):
        super().__init__('wall_right_up', color)
    
    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def can_go_through(self, direction):
        if direction != 0 and direction != 3:
            return True
        return False
    
    def can_enter_from(self, direction):
        if direction != 2 and direction != 1: #looking left or down
            return True
        return False
    
    def render(self, img):
        """
        x-axis is from left to right.
        y-axis is from top to bottom.
        """
        #fill_coords(img, point_in_rect(0.75, 1, 0.25, 0.75), COLORS[self.color])
        #fill_coords(img, point_in_rect(0.25, 0.75, 0, 0.25), COLORS[self.color])
        fill_coords(img, point_in_rect(0.75, 1, 0, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 1, 0, 0.25), COLORS[self.color])

class WallRightDown(CoMazeWall):
    def __init__(self, color='grey'):
        super().__init__('wall_right_down', color)
    
    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def can_go_through(self, direction):
        if direction != 0 and direction != 1:
            return True
        return False
    
    def can_enter_from(self, direction):
        if direction != 2 and direction != 3: #looking left or up
            return True
        return False
    
    def render(self, img):
        """
        x-axis is from left to right.
        y-axis is from top to bottom.
        """
        #fill_coords(img, point_in_rect(0.75, 1, 0.25, 0.75), COLORS[self.color])
        #fill_coords(img, point_in_rect(0.25, 0.75, 0.75, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0.75, 1, 0, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0, 1, 0.75, 1), COLORS[self.color])
        

class TimeBonus(CoMazeWorldObj):
    def __init__(self, color='grey'):
        super().__init__('time_bonus', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(
            img, 
            point_in_triangle(
                (0.19, 0.12),
                (0.50, 0.50),
                (0.81, 0.12),
            ), 
            COLORS[self.color]
        )
        fill_coords(
            img, 
            point_in_triangle(
                (0.19, 0.88),
                (0.50, 0.50),
                (0.81, 0.88),
            ), 
            COLORS[self.color]
        )


class CoMazeGoal(CoMazeWorldObj):
    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class RedGoal(CoMazeGoal):
    def __init__(self):
        super().__init__('red_goal', 'red')

class YellowGoal(CoMazeGoal):
    def __init__(self):
        super().__init__('yellow_goal', 'yellow')

class BlueGoal(CoMazeGoal):
    def __init__(self):
        super().__init__('blue_goal', 'blue')

class GreenGoal(CoMazeGoal):
    def __init__(self):
        super().__init__('green_goal', 'green')



class CoMazeGrid(Grid):
    def is_inside(self, i, j):
        xcond = i >= 0 and i < self.width
        ycond = j >= 0 and j < self.height
        return xcond and ycond

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w, obj_type=WallUp)
        self.horz_wall(x, y+h-1, w, obj_type=WallDown)
        self.vert_wall(x, y, h, obj_type=WallLeft)
        self.vert_wall(x+w-1, y, h, obj_type=WallRight)
        self.set(x, y, WallLeftUp())
        self.set(x+w-1, y, WallRightUp())
        self.set(x+w-1, y+h-1, WallRightDown())
        self.set(x, y+h-1, WallLeftDown())
        
    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = CoMazeGrid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = CoMazeGrid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def obj_decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        elif obj_type == 'wall_up':
            v = WallUp(color)
        elif obj_type == 'wall_down':
            v = WallDown(color)
        elif obj_type == 'wall_left':
            v = WallLeft(color)
        elif obj_type == 'wall_left_up':
            v = WallLeftUp(color)
        elif obj_type == 'wall_left_down':
            v = WallLeftDown(color)
        elif obj_type == 'wall_right':
            v = WallRight(color)
        elif obj_type == 'wall_right_up':
            v = WallRightUp(color)
        elif obj_type == 'wall_right_down':
            v = WallRightDown(color)
        elif obj_type == 'red_goal':
            v = RedGoal()
        elif obj_type == 'blue_goal':
            v = BlueGoal()
        elif obj_type == 'yellow_goal':
            v = YellowGoal()
        elif obj_type == 'green_goal':
            v = GreenGoal()
        elif obj_type == 'time_bonus':
            v = TimeBonus()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=np.bool)

        grid = CoMazeGrid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = CoMazeGrid.obj_decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            """
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))
            """
            draw_agent_fn = point_in_circle(cx=0.5, cy=0.50, r=0.25)
            fill_coords(img, draw_agent_fn, (255, 255, 255))
            
        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        # No highlighting needed since no direction:
        #if highlight_mask is None:
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = CoMazeGrid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img


from .comaze import Direction, CoMazeObject, Int2D, Goal, SecretGoalRule, Move

class CoMazeLocalGymEnv(MiniGridEnv):
    """
    """

    metadata = {
        'render.modes': ['human_comm', 'human', 'rgb_array'],
        'video.frames_per_second' : 2
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Move left, move right, move up, move down, skip
        left = 0
        right = 1
        up = 2
        down = 3
        skip = 4

    def __init__(
        self,
        width=None,
        height=None,
        see_through_walls=True,
        seed=1337,
        agent_view_size=7,
        sparse_reward=False,
        with_penalty=True,
        max_sentence_length=1,
        vocab_size=10,
        fixed_action_space=False,
        fixed_secret_goal_rule=False,
        single_player=False,
        timestep_increment=20,
        overall_max_steps=200, #400
        limit_step=True,
        joint_reward=True,
        reset_level=1,
        gameover_on_rule_breaching=False,
        secret_goal_rule_breaching_penalty=-1,
        secret_goal_rule_selection="normal", #easy/hard/normal/uniform
        #secret_goal_rule_breaching_penalty=-1,
        goal_reaching_reward=1,
    ):  
        self.joint_reward = joint_reward

        self.rotatable_view = False
        self.with_penalty = with_penalty
        self.fixed_action_space = fixed_action_space
        self.fixed_secret_goal_rule = fixed_secret_goal_rule
        self.secret_goal_rule_selection = secret_goal_rule_selection
        self.gameover_on_rule_breaching = gameover_on_rule_breaching
        self.goal_reaching_reward = goal_reaching_reward
        self.secret_goal_rule_breaching_penalty = secret_goal_rule_breaching_penalty
        self.single_player = single_player
        self.timestep_increment = timestep_increment
        
        self.limit_step = limit_step
        self.overall_max_steps_init = overall_max_steps
        self.overall_max_steps = overall_max_steps
        self.step_count = 0

        self.level = 1 
        self.reset_level = reset_level
        self.nbr_players = 2

        self.secretgoalEnum2id = {"red_goal":0, "yellow_goal":1, "blue_goal":2, "green_goal":3}
        self.id2SecretgoalEnum = dict(zip(self.secretgoalEnum2id.values(), self.secretgoalEnum2id.keys()))
        self.secretgoalStr2id = {"RED":0, "YELLOW":1, "BLUE":2, "GREEN":3}
        self.id2SecretgoalStr = dict(zip(self.secretgoalStr2id.values(), self.secretgoalStr2id.keys()))
        self.actionId2Str = {0:"LEFT", 1:"RIGHT", 2:"UP", 3:"DOWN", 4:"SKIP"}

        self.sparse_reward = sparse_reward

        self.agent_start_pos = (width//2, height//2)
        self.agent_start_dir = 0

        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size

        # Directional action enumeration for this environment
        self.directional_actions = CoMazeLocalGymEnv.Actions
        
        # Actions consist of a dictionnary of two elements:
        # - directional actions that are discrete integer values
        # - communication channel that consist of ungrounded tokdict(zip(self.secretgoalEnum2id.values(), self.secretgoalEnum2id.keys()))ens, represented as integer values.
        self.directional_action_space = spaces.Discrete(len(self.directional_actions))
        self.communication_channel_action_space = CommunicationChannel(
            max_sentence_length=self.max_sentence_length,
            vocab_size=self.vocab_size
        )
        self.action_space = spaces.Dict({
            'directional_action': self.directional_action_space,
            'communication_channel': self.communication_channel_action_space
        })
        # 2 players: unnecessary...
        # self.action_space = spaces.Tuple([self.action_space for _ in range(self.nbr_players)])

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing:
        # -an encoding of the grid,
        # -a communication channel output.
        # -a list of available directional actions.
        # -a secret goal rule.
        self.grid_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.communication_channel_observation_space = copy.deepcopy(self.communication_channel_action_space)

        self.available_directional_actions_observation_space = spaces.MultiBinary(n=5) #4 direction and skip

        # 4 possible goals:
        #  earlierGoal and laterGoal are encoded as a MultiBinary(n=4*2), 
        #  the first 4 binary one-hot encode the earlierGoal 
        #  while the following 4 one-hot encode the laterGoal: 
        self.secret_goal_rule_observation_space = spaces.MultiBinary(n=4*2)

        self.agent_id_observation_space = spaces.MultiBinary(n=self.nbr_players)

        self.observation_space = spaces.Dict({
            'image': self.grid_observation_space,
            'communication_channel': self.communication_channel_observation_space,
            'available_directional_actions': self.available_directional_actions_observation_space,
            'secret_goal_rule': self.secret_goal_rule_observation_space,
            'agent_id': self.agent_id_observation_space,
        })
        # 2 players: unnecessary...
        #self.observation_space = spaces.Tuple([self.observation_space for _ in range(self.nbr_players)])

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self, level=None):
        if level is None:
            level =self.reset_level

        self.level = level
        self.done = False 

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        self.nbr_reached_goals = 0
        self.new_goal_reached = False 

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Secret Goal Rules:
        if self.secret_goal_rule_selection == 'uniform':
            types = ['easy','hard']
            secret_goal_rule_selection = types[np.random.randint(2)]
        else:
            secret_goal_rule_selection = self.secret_goal_rule_selection

        self.secret_goal_rules = list()
        self.secretGoalRuleObjects = []
        conflict = self.level >= 4 
        while conflict:
            if secret_goal_rule_selection == "hard":
                # exactly one goal in common 
                # at different position (earlier vs later):
                goal_choice_set = set(list(range(4)))
                commonGoal = np.random.randint(low=0,high=4)
                goal_choice_set.remove(commonGoal)
            elif secret_goal_rule_selection == "easy":
                # no goal in common
                goal_choice_set = set(list(range(4)))

            earlierLaterGoals = []
            for pidx in range(self.nbr_players):
                if self.level>=4:
                    if self.fixed_secret_goal_rule:
                        assert self.nbr_players == 2
                        if pidx==0:
                            earlierLaterGoals.append([2,0])
                        else:
                            earlierLaterGoals.append([3,1])
                    else:

                        if secret_goal_rule_selection == "normal":
                            earlierLaterGoals.append( np.random.choice(
                                    a=np.arange(4), 
                                    size=2,
                                    replace=False,
                                )
                            )
                        elif secret_goal_rule_selection == "hard":
                            assert self.nbr_players == 2 
                            chosen = random.choice(list(goal_choice_set))
                            goal_choice_set.remove(chosen) # prevent conflict...
                            rule = np.zeros(2, dtype=np.int32)
                            if pidx %2:
                                rule[0] = commonGoal
                                rule[1] = chosen 
                            else:
                                rule[0] = chosen
                                rule[1] = commonGoal 
                            earlierLaterGoals.append(rule)
                        elif secret_goal_rule_selection == "easy":
                            assert self.nbr_players == 2 
                            goal1 = random.choice(list(goal_choice_set))
                            goal_choice_set.remove(goal1)
                            goal2 = random.choice(list(goal_choice_set))
                            goal_choice_set.remove(goal2)
                            rule = np.zeros(2, dtype=np.int32)
                            rule[0] = goal1
                            rule[1] = goal2 
                            earlierLaterGoals.append(rule)

            # is it valid?
            if self.level >= 4:
                assert self.nbr_players == 2 
                conflict = (earlierLaterGoals[pidx][0] == earlierLaterGoals[1-pidx][1]) and  (earlierLaterGoals[pidx][1] == earlierLaterGoals[1-pidx][0])


        for player_idx in range(self.nbr_players):
            secretGoalRule = np.zeros((1, 8))
            if self.level>=4:
                secretGoalRule[0, earlierLaterGoals[player_idx][0]] = 1
                secretGoalRule[0, 4+earlierLaterGoals[player_idx][1]] = 1

                eg = earlierLaterGoals[player_idx][0]
                lg = earlierLaterGoals[player_idx][1]
                eg_color = self.id2SecretgoalStr[eg]
                lg_color = self.id2SecretgoalStr[lg]
                self.secretGoalRuleObjects.append(
                    SecretGoalRule(
                        earlierGoal=Goal(
                            position=Int2D(
                                x=self.goal_positions[eg_color][0],
                                y=self.goal_positions[eg_color][1]
                            ), 
                            color=eg_color
                        ), 
                        laterGoal=Goal(
                            position=Int2D(
                                x=self.goal_positions[lg_color][0],
                                y=self.goal_positions[lg_color][1]
                            ), 
                            color=lg_color
                        ),
                    )
                )
            self.secret_goal_rules.append(secretGoalRule)

        # Available directional actions:
        self.available_directional_actions = list()
        if self.single_player:
            skip_action = np.asarray([[4]])
            for player_idx in range(self.nbr_players):
                if player_idx == 0:
                    ada = np.reshape(
                        np.arange(4),
                        (1, -1),
                    )
                    # adding skip:
                    ada = np.concatenate([ada, skip_action], axis=-1)
                else:
                    ada = skip_action    
                self.available_directional_actions.append(ada)
            
        else:
            nb_actions_per_player = 4 // self.nbr_players
            if self.fixed_action_space:
                availableDirectionalActions = np.reshape(
                    np.arange(4),
                    (self.nbr_players, nb_actions_per_player),
                )
            else:
                availableDirectionalActions = np.random.choice(
                    a=np.arange(4),
                    size=(self.nbr_players, nb_actions_per_player),
                    replace=False
                )
            self.availableDirectionalActions_indices = availableDirectionalActions    
            skip_action = np.asarray([[4]])
            for player_idx in range(self.nbr_players):
                ada = availableDirectionalActions[player_idx:player_idx+1] #shape 1 x nb_actions_per_player
                # adding skip:
                ada = np.concatenate([ada, skip_action], axis=-1)
                self.available_directional_actions.append(ada)

        # Reached goals:
        self.reached_goals = list()
        self.secret_goal_rule_breached = False 

        # Item picked up, being carried, initially nothing
        # not used in CoMaze but necessary for retro-compatibility with minigrid...
        self.carrying = None

        if self.level >= 2 and self.level != self.reset_level:
            self.overall_max_steps -= self.step_count
        else:
            self.overall_max_steps = self.overall_max_steps_init

        if self.level!=self.reset_level and self.level >= 3:
            # not enough time to go through all the goals, need to get time-bonuses:
            self.max_steps = 30
        else:
            self.max_steps = 100
        
        # Step count since episode start
        self.step_count = 0

        # Communication channel:
        self.communication_channel_content = np.zeros((1, self.max_sentence_length))
        if self.level==self.reset_level:
            self.communication_history = ['START']


        self.agent_ids = []
        for pidx in range(self.nbr_players):
            pidx_ohe = np.zeros((1,self.nbr_players))
            pidx_ohe[0, pidx] = 1
            self.agent_ids.append(pidx_ohe)

        #self.current_player = 0
        self.current_player = np.random.choice(list(range(self.nbr_players)))

        # Return first observation
        obs = self.gen_obs()

        info = {
            'current_player': np.ones((1,1), dtype=np.int32)*self.current_player,
        }

        # Update abstract repr:
        #self.lastAction = None
        self.lastMove = Move(
            action = None, 
            predicted_action = None, 
            symbol_message = None,
            predicted_goal = None
        )
        self.abstract_repr = {
            "arenaSize": copy.deepcopy(Int2D(self.height, self.width)),
            "agentPosition": copy.deepcopy(self.agent_pos),
            "goals": copy.deepcopy(self.goal_positions),
            "reached_goals": [ self.id2SecretgoalStr[self.secretgoalEnum2id[g]] for g in self.reached_goals],
            #"lastAction":
            "last_move":self.lastMove,
            "directions": [
                Direction("UP","UP", 0, -1),
                Direction("DOWN","DOWN", 0, 1),
                Direction("LEFT","LEFT", -1, 0),
                Direction("RIGHT","RIGHT", 1, 0)
            ],
            "actions": copy.deepcopy(self.availableDirectionalActions_indices),
            "secretGoalRule": copy.deepcopy(self.secretGoalRuleObjects),
            "current_player": self.current_player,
        }
        info['abstract_repr'] = self.abstract_repr

        infos = [copy.deepcopy(info) for _ in range(self.nbr_players)]
        for player_id in range(self.nbr_players):
            infos[player_id]["abstract_repr"]["player_id"] = player_id

        return obs, infos

    def _regularise_communication_channel(self, communication_channel_output):
        # Regularise the use of EoS symbol:
        make_eos = False
        # batch dim=1 x max_sentence_length...
        for idx, o in enumerate(communication_channel_output[0]):
            if make_eos:    
                communication_channel_output[0, idx] = 0
                continue
            if o==0:
                make_eos = True
        
        return communication_channel_output

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = CoMazeGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.goal_positions = {}

        if True: #self.level != 5:
            assert self.level == 5
            # SQUARE:
            # Place red goal square:
            self.put_obj(RedGoal(), 1,1)
            self.goal_positions['RED'] = (1,1)
            # Place yellow goal square:
            self.put_obj(YellowGoal(), width-2,1)
            self.goal_positions['YELLOW'] = (width-2,1)
            # Place green goal square:
            self.put_obj(GreenGoal(), 1,height-2)
            self.goal_positions['GREEN'] = (1,height-2)
            # Place blue goal square:
            self.put_obj(BlueGoal(), width-2,height-2)
            self.goal_positions['BLUE'] = (width-2,height-2)

            # PRISM:
            # # Place red goal square:
            # self.put_obj(RedGoal(), 1,height//2)
            # self.goal_positions['RED'] = (1,height//2)
            # # Place yellow goal square:
            # self.put_obj(YellowGoal(), width//2,1)
            # self.goal_positions['YELLOW'] = (width//2,1)
            # # Place green goal square:
            # self.put_obj(GreenGoal(), width-2,height//2)
            # self.goal_positions['GREEN'] = (width-2,height//2)
            # # Place blue goal square:
            # self.put_obj(BlueGoal(), width//2,height-2)
            # self.goal_positions['BLUE'] = (width//2,height-2)
        else:
            raise NotImplementedError
            # Place red goal square:
            self.put_obj(RedGoal(), 2,2)
            # Place yellow goal square:
            self.put_obj(YellowGoal(), width-3,2)
            # Place green goal square:
            self.put_obj(GreenGoal(), 2,height-3)
            # Place blue goal square:
            self.put_obj(BlueGoal(), width-3,height-3)
        
        """
        if self.level >= 2:
            self.put_obj(WallRight(), 2, 2)
            self.put_obj(WallRight(), 2, 4)
            self.put_obj(WallRight(), 2, 5)
            self.put_obj(WallRightDown(), 5, 2)
            self.put_obj(WallRightDown(), 4, 3)
            self.put_obj(WallDown(), 3, 2)
        """
        if self.level >= 2 and self.level != 5:
            self.put_obj(Wall(), 2, 2)
            self.put_obj(Wall(), 2, 4)
            self.put_obj(Wall(), 2, 5)
            self.put_obj(Wall(), 5, 2)
            self.put_obj(Wall(), 4, 3)
            self.put_obj(Wall(), 3, 2)
                
        if self.level >= 3 and self.level !=5:
            self.put_obj(TimeBonus(), 3, 0)
            self.put_obj(TimeBonus(), 3, 6)
            self.put_obj(TimeBonus(), 0, 3)
            self.put_obj(TimeBonus(), 6, 3)


        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def gen_obs_grid(self):
        """
        Deprecated for full-observability...

        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        if self.rotatable_view:
            for i in range(self.agent_dir + 1):
                grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        """
        # Deprecated for full observability:
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)
        """
        image = self.grid.encode()

        # Observations are dictionaries containing:
        # -an encoding of the grid,
        # -a communication channel output.
        # -a list of available directional actions.
        # -a secret goal rule.       
        p1_obs = {
            'image': image,
            'communication_channel': self.communication_channel_content,
            'available_directional_actions': self.available_directional_actions[0],
            'secret_goal_rule': self.secret_goal_rules[0],
            'agent_id': self.agent_ids[0],
        }

        p2_obs = {
            'image': image,
            'communication_channel': self.communication_channel_content,
            'available_directional_actions': self.available_directional_actions[1],
            'secret_goal_rule': self.secret_goal_rules[1],
            'agent_id': self.agent_ids[1],
        }

        return [p1_obs, p2_obs]

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = CoMazeGrid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'F',
            'door'          : 'D',
            'key'           : 'K',
            'ball'          : 'A',
            'box'           : 'B',
            'goal'          : 'G',
            'lava'          : 'V',
            'wall_up'       : ' ',
            'wall_down'     : '_',
            'wall_left'     : '|',
            'wall_left_up'  : '[',
            'wall_left_down': ']',
            'wall_right'    : '|',
            'wall_right_up' : '<',
            'wall_right_down': '>',
            'time_bonus'    : 'T',
            'red_goal'      : 'G',
            'blue_goal'     : 'G',
            'yellow_goal'   : 'G',
            'green_goal'    : 'G',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 'AA'
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _can_go_through(self, cell, direction):
        if not isinstance(cell, CoMazeWall):
            return True

        return cell.can_go_through(direction=direction)

    def _can_enter_from(self, cell, direction):
        if not isinstance(cell, CoMazeWall):
            return True

        return cell.can_enter_from(direction=direction)

    def _secret_goal_rule_breached(self):
        breached = False
        reached_goalsIds = []
        for reachedGoal in self.reached_goals:
            reached_goalsIds.append(self.secretgoalEnum2id[reachedGoal]) 

        for player_idx, secretGoalRule in enumerate(self.secret_goal_rules):
            if secretGoalRule.sum()==0:   continue
            earlierGoal = secretGoalRule[0, :4].argmax(axis=0)
            laterGoal = secretGoalRule[0, 4:].argmax(axis=0)
            # We only care about the latest goal that has been newly reached:
            # Doing so, we can continue the game without receiving penalties
            # at every steps. 
            if laterGoal==reached_goalsIds[-1] \
            and earlierGoal not in reached_goalsIds:
                breached = True 
                break
        self.secret_goal_rule_breached = breached 
        return breached 

    def step(self, action):
        self.step_count += 1

        reward = 0
        assert self.done==False, "Please reset this environment, it has terminated."

        directional_action = action.get("directional_action", 4) #skip if not there...        
        communication_channel_output = action.get("communication_channel", np.zeros(shape=(1, self.max_sentence_length), dtype=np.int64))

        # Communication Channel:
        reg_communication_channel_output = self._regularise_communication_channel(communication_channel_output=communication_channel_output)
        self.communication_channel_content = reg_communication_channel_output
        if not hasattr(self, 'communication_history'):  self.communication_history = ['START']
        message = f"P{self.current_player+1}:"
        if self.level >= 4:
            earlierGoal = self.secret_goal_rules[self.current_player][0, :4].argmax(axis=0)
            laterGoal = self.secret_goal_rules[self.current_player][0, 4:].argmax(axis=0)
            message += self.id2SecretgoalEnum[earlierGoal][0]+self.id2SecretgoalEnum[laterGoal][0]
        message += f":{self.communication_channel_content[0]}"
        self.communication_history.append(message)

        # Directional action:
        directional_move = True
        if directional_action not in self.available_directional_actions[self.current_player]\
         or directional_action==self.directional_actions.skip:
            directional_move = False
        
        if directional_move:
            # Facing right and ...
            if directional_action == self.directional_actions.right:
                self.agent_dir = 0  
            # Facing left and ...
            elif directional_action == self.directional_actions.left:
                self.agent_dir = 2
            # Facing up and ...
            elif directional_action == self.directional_actions.up:
                self.agent_dir = 3  
            # Facing down and ...
            elif directional_action == self.directional_actions.down:
                self.agent_dir = 1
            else:            
                assert directional_action==4, "invalid agent direction"
                directional_move = False

            # Get the position in front of the agent, 
            # from a property of MiniGridEnv that infers 
            # the forward position using the agent_dir value
            # that has just been set, above: 
            fwd_pos = self.front_pos

            #If fwd_pos is no longer inside the grid:
            if self.grid.is_inside(*fwd_pos):
                # Get the contents of the cell in front of the agent
                fwd_cell = self.grid.get(*fwd_pos)
                # Get the content of the cell where the agent is:
                current_cell = self.grid.get(*self.agent_pos)
                
                # ... check that there is no wall in the forward direction...
                if self._can_go_through(cell=current_cell, direction=self.agent_dir) \
                and self._can_enter_from(cell=fwd_cell, direction=self.agent_dir):
                    # ... move forward
                    if fwd_cell == None or fwd_cell.can_overlap():
                        self.agent_pos = fwd_pos

                    # Have we reached a new goal?
                    if fwd_cell != None and 'goal' in fwd_cell.type \
                    and fwd_cell.type not in self.reached_goals:
                        self.new_goal_reached = True
                        # Make the reached new goal grey:
                        fwd_cell.color = 'black'
                        #since it is a reference, no need for the following:
                        #self.grid.set(*fwd_pos, fwd_cell)
                        # Record the recently reached new goal:
                        if fwd_cell.type not in self.reached_goals:
                            self.reached_goals.append(fwd_cell.type)

                    # Have we reached a time bonus?
                    if fwd_cell != None and 'time_bonus' in fwd_cell.type:
                        self.max_steps += self.timestep_increment
                else:
                    # silent failure when the directional action is not executable due to a CoMazeWall.
                    #print("silent failure when the directional action is not executable.")
                    pass
            else:
                # failure when the directional action is not executable due to a the fwd pos being 
                # outside of the grid.
                directional_move = False 

        # Can the game still carry on?
        if directional_move:
            if fwd_cell != None and fwd_cell.type == 'lava':
                self.done = True
            # Has any any secret goal rule been breached:
            if self.new_goal_reached and self._secret_goal_rule_breached() and self.gameover_on_rule_breaching:
                self.done = True
        if self.step_count >= self.max_steps or (self.limit_step and self.step_count > self.overall_max_steps):
            self.done = True

        # Reward computation:
        reward = self._reward()
        if self.joint_reward:
            reward_vector = [reward for _ in range(self.nbr_players)]
        else:
            reward_vector = [0.0 for _ in range(self.nbr_players)]
            reward_vector[self.current_player] = reward
        
        # Carry on to the next level 
        # if all goals have been reached:
        if self.nbr_reached_goals==4:
            if self.level<4:
                obs, _ = self.reset(level=self.level+1)
            else:
                self.done = True
                obs = self.gen_obs()
        else:
            obs = self.gen_obs()


        #Bookkeeping:
        # regularising for rendering gen_obs_grid fn...
        self.agent_dir = 0

        self.current_player = (self.current_player+1)%self.nbr_players
        
        info = {
            'current_player': np.ones((1,1), dtype=np.int32)*self.current_player,
        }
        

        # Update abstract repr:
        #self.lastAction = None
        self.lastMove = Move(
            action = self.actionId2Str[directional_action], 
            predicted_action = None, 
            predicted_goal = None,
            symbol_message = int(reg_communication_channel_output.item()) ,
        )
        self.abstract_repr = {
            "arenaSize": copy.deepcopy(Int2D(self.height, self.width)),
            "agentPosition": copy.deepcopy(self.agent_pos),
            "goals": copy.deepcopy(self.goal_positions),
            "reached_goals": [ self.id2SecretgoalStr[self.secretgoalEnum2id[g]] for g in self.reached_goals],
            #"lastAction":
            "last_move":self.lastMove,
            "directions": [
                Direction("UP","UP", 0, -1),
                Direction("DOWN","DOWN", 0, 1),
                Direction("LEFT","LEFT", -1, 0),
                Direction("RIGHT","RIGHT", 1, 0)
            ],
            "actions": copy.deepcopy(self.availableDirectionalActions_indices),
            "secretGoalRule": copy.deepcopy(self.secretGoalRuleObjects),
            "current_player": self.current_player,
        }
        info['abstract_repr'] = self.abstract_repr

        """
        [trajectory[idx][-1][-1]["abstract_repr"]['secretGoalRule'][0].earlierGoal.color, trajectory[idx][-1][-1]["abstract_repr"]['secretGoalRule'][0].laterGoal.color, 
        trajectory[idx][-1][-1]["abstract_repr"]['secretGoalRule'][1].earlierGoal.color, trajectory[idx][-1][-1]["abstract_repr"]['secretGoalRule'][1].laterGoal.color]
        
        trajectory[idx][-1][-1]["abstract_repr"]['reached_goals']
        
        [info[0][actor_index]["abstract_repr"]['secretGoalRule'][0].earlierGoal.color, info[0][actor_index]["abstract_repr"]['secretGoalRule'][0].laterGoal.color, 
        info[0][actor_index]["abstract_repr"]['secretGoalRule'][1].earlierGoal.color, info[0][actor_index]["abstract_repr"]['secretGoalRule'][1].laterGoal.color]
        """
        infos = [copy.deepcopy(info) for _ in range(self.nbr_players)]
        #infos[1-self.current_player]["abstract_repr"]["current_player"] = self.current_player
        for player_id in range(self.nbr_players):
            infos[player_id]["abstract_repr"]["player_id"] = player_id
            
        return obs, reward_vector, self.done, infos

    def _reward(self):
        reward = 0
        if self.new_goal_reached:
            self.nbr_reached_goals += 1

        if self.sparse_reward:
            if self.nbr_reached_goals==4:
                reward = 1
            elif self.done:
                reward = -1
    
        if not self.sparse_reward:
            if self.new_goal_reached:
                reward += self.goal_reaching_reward
            elif self.with_penalty:
                reward -= 0.1
            else:
                pass

        if self.secret_goal_rule_breached:
            self.secret_goal_rule_breached = False 
            reward = self.secret_goal_rule_breaching_penalty

        # Bookkeeping:
        if self.new_goal_reached:
            self.new_goal_reached = False 
            
        return reward 

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window(f"CoMaze-level:{self.level}")
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(f"Communication Channel: {self.communication_channel_content}")
        elif mode == 'human_comm':
            img = np.concatenate([img, np.zeros_like(img)], axis=1)
            orig_x = img.shape[0]
            orig_y = img.shape[1]
            
            y_inc = (orig_y-20)//8
            for idx in range(4):
                comm = ''
                for commidx in range(2):
                    if 2*idx+commidx+1>=len(self.communication_history): 
                        break
                    comm += ' ' + self.communication_history[-(2*idx+commidx+1)]
                position = (orig_x//2+150, 30+y_inc*idx)
                cv2.putText(
                    img,
                    comm,
                    position, #position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, #font family
                    0.5, #font size
                    (20, 20, 255, 255), #font color
                    2,  #stroke
                )
                if idx>=len(self.communication_history):
                    break
            """
            comm = ''
            for idx in range(8):
                comm += ' ' + self.communication_history[-(idx+1)]
                if idx%2:
                    comm += '\n'
                if idx+1>=len(self.communication_history):
                    break

            position = (orig_x//2+80, 30)
            cv2.putText(
                img,
                comm,
                position, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.5, #font size
                (20, 20, 255, 255), #font color
                2,  #stroke
            )
            """
            
            """
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
            plt.savefig('test2.png')
            import ipdb; ipdb.set_trace()
            """             

        return img


class CoMazeGymEnv7x7Sparse(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=True,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            **kwargs
        )

class CoMazeGymEnv11x11Sparse(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=11,
            height=11,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=True,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = CoMazeGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place red goal square:
        self.put_obj(RedGoal(), 2,2)
        # Place yellow goal square:
        self.put_obj(YellowGoal(), width-3,2)
        # Place green goal square:
        self.put_obj(GreenGoal(), 2,height-3)
        # Place blue goal square:
        self.put_obj(BlueGoal(), width-3,height-3)
        
        """
        if self.level >= 2:
            self.put_obj(WallRight(), 2, 2)
            self.put_obj(WallRight(), 2, 4)
            self.put_obj(WallRight(), 2, 5)
            self.put_obj(WallRightDown(), 5, 2)
            self.put_obj(WallRightDown(), 4, 3)
            self.put_obj(WallDown(), 3, 2)
        """
        if self.level >= 2:
            self.put_obj(Wall(), 3, 3)
            self.put_obj(Wall(), 3, 5)
            self.put_obj(Wall(), 3, 6)
            self.put_obj(Wall(), 6, 3)
            self.put_obj(Wall(), 5, 4)
            self.put_obj(Wall(), 4, 3)

            self.put_obj(Wall(), 7, 3)
            self.put_obj(Wall(), 7, 2)
            self.put_obj(Wall(), 3, 8)
            self.put_obj(Wall(), 4, 8)


            self.put_obj(Wall(), 8, 9)
            self.put_obj(Wall(), 7, 8)
                
        if self.level >= 3:
            self.put_obj(TimeBonus(), 5, 0)
            self.put_obj(TimeBonus(), 5, 10)


        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


class CoMazeGymEnv11x11Dense(CoMazeGymEnv11x11Sparse):
    def __init__(self, **kwargs):
        CoMazeLocalGymEnv.__init__(
            self,
            width=11,
            height=11,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            **kwargs
        )

class CoMazeGymEnv7x7Dense(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            **kwargs
        )

class CoMazeGymEnv7x7DenseSinglePlayerReward(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            joint_reward=False,
            **kwargs
        )

class CoMazeGymEnv7x7DenseLevel4(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=True, #False,
            max_sentence_length=1,
            vocab_size=3, #20,
            joint_reward=True,
            reset_level=4,
            fixed_secret_goal_rule=False,
            overall_max_steps=50,#100,
            secret_goal_rule_breaching_penalty=-1,
            goal_reaching_reward=1,
            **kwargs
        )


class CoMazeGymEnv7x7DenseLevel5(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty= True, #False,
            max_sentence_length=1,
            vocab_size=20, #10,
            joint_reward=True,
            reset_level=5,
            fixed_secret_goal_rule=False,
            overall_max_steps=50,#100,
            secret_goal_rule_breaching_penalty=-1,
            goal_reaching_reward=1,
            **kwargs
        )

class CoMazeGymEnv9x9DenseLevel5(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=9,
            height=9,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty= True, #False,
            max_sentence_length=1,
            vocab_size=20, #10,
            joint_reward=True,
            reset_level=5,
            fixed_secret_goal_rule=False,
            overall_max_steps=100,
            secret_goal_rule_breaching_penalty=-1,
            secret_goal_rule_selection="normal",
            goal_reaching_reward=1,
            **kwargs
        )

class CoMazeGymEnv9x9DenseLevel5EasySecrets(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=9,
            height=9,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty= True, #False,
            max_sentence_length=1,
            vocab_size=20, #10,
            joint_reward=True,
            reset_level=5,
            fixed_secret_goal_rule=False,
            overall_max_steps=100,
            secret_goal_rule_breaching_penalty=-1,
            secret_goal_rule_selection="easy",
            goal_reaching_reward=1,
            **kwargs
        )

class CoMazeGymEnv9x9DenseLevel5HardSecrets(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=9,
            height=9,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty= True, #False,
            max_sentence_length=1,
            vocab_size=20, #10,
            joint_reward=True,
            reset_level=5,
            fixed_secret_goal_rule=False,
            overall_max_steps=100,
            secret_goal_rule_breaching_penalty=-5,
            secret_goal_rule_selection="hard",
            goal_reaching_reward=1,
            **kwargs
        )

class CoMazeGymEnv9x9DenseLevel5UniformSecrets(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=9,
            height=9,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty= True, #False,
            max_sentence_length=1,
            vocab_size=20, #10,
            joint_reward=True,
            reset_level=5,
            fixed_secret_goal_rule=False,
            overall_max_steps=100,
            secret_goal_rule_breaching_penalty=-1,
            secret_goal_rule_selection="uniform",
            goal_reaching_reward=1,
            **kwargs
        )

class CoMazeGymEnv7x7DenseSinglePlayerRewardLevel4(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            joint_reward=False,
            reset_level=4,
            fixed_secret_goal_rule=False,
            **kwargs
        )

class CoMazeGymEnv7x7DenseSinglePlayerRewardLevel4FixedSecretGoalRules(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            joint_reward=False,
            reset_level=4,
            fixed_secret_goal_rule=True,
            **kwargs
        )

class CoMazeGymEnv7x7DenseFixedActions(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            fixed_action_space=True,
            **kwargs
        )


class CoMazeGymEnv7x7DenseEasySinglePlayer(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            single_player=True,
            timestep_increment=100,
            **kwargs
        )

class CoMazeGymEnv7x7DenseSinglePlayer(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=False,
            max_sentence_length=1,
            vocab_size=10,
            single_player=True,
            **kwargs
        )

class CoMazeGymEnv7x7DenseWithPenalty(CoMazeLocalGymEnv):
    def __init__(self, **kwargs):
        super().__init__(
            width=7,
            height=7,
            see_through_walls=True,
            seed=1337,
            agent_view_size=7,
            sparse_reward=False,
            with_penalty=True,
            max_sentence_length=1,
            vocab_size=10,
            **kwargs
        )
