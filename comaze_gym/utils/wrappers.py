import copy
import numpy as np
import gym
import cv2 

from gym_minigrid.minigrid import *

class Bijection(object):
    def __init__(self, env, vocab_only=True):
        self.env = env 
        self.vocab_only = vocab_only

        self.vocab_size = self.env.vocab_size
        self.max_sentence_length = self.env.max_sentence_length
        
        shuffledarr = np.arange(start=1,stop=self.vocab_size+1)
        np.random.shuffle(shuffledarr)
        self.communication_channel_bijection_decoder = { idx+1: v.item() for idx, v in enumerate(shuffledarr)}
        self.communication_channel_bijection_decoder[0] = 0 
        self.communication_channel_bijection_encoder = dict(zip(self.communication_channel_bijection_decoder.values(), self.communication_channel_bijection_decoder.keys()))

        self.direction_action_bijection_decoder = { 
            "identity": {v: v for v in range(5)},
            "vertical_mirror": {0:0, 1:1, 2:3, 3:2, 4:4},
            "vertical_horizontal_mirror": {0:1, 1:0, 2:3, 3:2, 4:4},
            "horizontal_mirror": {0:1, 1:0, 2:2, 3:3, 4:4},
        }
        self.direction_action_bijection_encoder = { 
            "identity": {v: v for v in range(5)},
            "horizontal_mirror": dict(
                zip(
                    self.direction_action_bijection_decoder["horizontal_mirror"].values(), 
                    self.direction_action_bijection_decoder["horizontal_mirror"].keys()
                )
            ),
            "vertical_horizontal_mirror": dict(
                zip(
                    self.direction_action_bijection_decoder["vertical_horizontal_mirror"].values(), 
                    self.direction_action_bijection_decoder["vertical_horizontal_mirror"].keys()
                )
            ),
            "vertical_mirror": dict(
                zip(
                    self.direction_action_bijection_decoder["vertical_mirror"].values(), 
                    self.direction_action_bijection_decoder["vertical_mirror"].keys()
                )
            ),
        }
        self.pixel_obs_bijection_encoder = {
            "identity": lambda x: x,
            "vertical_mirror": (lambda x: cv2.flip(x, 0)),
            "horizontal_mirror": (lambda x: cv2.flip(x, 1)),
            "vertical_horizontal_mirror": (lambda x: cv2.flip(cv2.flip(x, 1), 0)),
        }
        self.bijectionIntToStr = {
            0:"identity",
            1:"vertical_mirror",
            2:"horizontal_mirror",
            3:"vertical_horizontal_mirror",
        }
        self.bijection_str = None
        # TODO: try with rotational bijections...
        # TODO: try changing goal's color (see secret goal rule too...)

    def reset(self):

        # Directional Action:
        bijection_idx = np.random.randint(4)
        if self.vocab_only: bijection_idx = 0
        self.bijection_str = self.bijectionIntToStr[bijection_idx]
        
        # Communication Channel:
        shuffledarr = np.arange(start=1,stop=self.vocab_size+1)
        np.random.shuffle(shuffledarr)
        self.communication_channel_bijection_decoder = { idx+1: v.item() for idx, v in enumerate(shuffledarr)}
        self.communication_channel_bijection_decoder[0] = 0 
        self.communication_channel_bijection_encoder = dict(zip(self.communication_channel_bijection_decoder.values(), self.communication_channel_bijection_decoder.keys()))
        

    def encode_obs(self, obs):
        """

        """
        self.previous_obs = copy.deepcopy(obs)

        image = obs["image"]
        image = self.pixel_obs_bijection_encoder[self.bijection_str](image)
        obs["image"] = image

        ada = obs["available_directional_actions"]
        for adaidx in range(ada.shape[-1]):
            ada[0, adaidx] = self.direction_action_bijection_encoder[self.bijection_str][ada[0,adaidx]]
        obs["available_directional_actions"] = ada

        comm = obs["communication_channel"]
        for idx in range(self.max_sentence_length):
            comm[idx] = self.communication_channel_bijection_encoder[comm[idx].item()]
        obs["communication_channel"] = comm
        
        return obs

    def decode_action(self, action):
        """
        :param Action: Dict that contains the keys:
            - "directional_action": Int in range [0,4]
            - "communication_channel": ... 
        """
        self.previous_action = copy.deepcopy(action)

        dir_action = action.get("directional_action")
        dir_action = self.direction_action_bijection_decoder[self.bijection_str][dir_action]
        action["directional_action"] = dir_action

        # Communication Channel:
        comm = action.get("communication_channel", np.zeros(shape=(1, self.max_sentence_length), dtype=np.int64))
        for idx in range(self.max_sentence_length):
            comm[idx] = self.communication_channel_bijection_decoder[comm[idx].item()]
        action["communication_channel"] = comm 

        return action


class RGBImgWrapper(gym.Wrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype='uint8'
        )

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        env = self.unwrapped
        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )
        for player_idx, obs in enumerate(observations):
            observations[player_idx]["image"] = rgb_img

        return observations, infos

    def step(self, action):
        next_observations, reward, done, next_infos = self.env.step(action)
        env = self.unwrapped
        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )
        for player_idx, obs in enumerate(next_observations):
            next_observations[player_idx]["image"] = rgb_img

        return next_observations, reward, done, next_infos

class DiscreteCombinedActionWrapper(gym.Wrapper):
    """
    Assumes the :arg env: environment's action space is a Dict that contains 
    the keys "communication_channel" and "directional_action".
    Firstly, it combines both spaces into a Discrete action space, and adds a No-op action
    at the end for the player who cannot play the current turn.
    Secondly, it augments the infos list of dictionnary with entries 
    "legal_actions" and "action_mask", for each player's info. 
    Args:
        - env (gym.Env): Env to wrap.
    """
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Move left, move right, move up, move down, skip
        left = 0
        right = 1
        up = 2
        down = 3
        skip = 4

    def __init__(self, env):
        super(DiscreteCombinedActionWrapper, self).__init__(env)
        self.wrapped_action_space = env.action_space 
        
        self.vocab_size = self.wrapped_action_space.spaces["communication_channel"].vocab_size
        self.max_sentence_length = self.wrapped_action_space.spaces["communication_channel"].max_sentence_length

        self.nb_directions = self.wrapped_action_space.spaces["directional_action"].n 
        
        self._build_sentenceId2sentence()
        
        # Action Space:
        self.nb_possible_actions = self.nb_directions*self.nb_possible_sentences
        # Adding no-op action:
        self.action_space = gym.spaces.Discrete(self.nb_possible_actions+1)

        self.observation_space = env.observation_space

    def _build_sentenceId2sentence(self):
        self.nb_possible_sentences = 1 # empty string...
        for pos in range(self.max_sentence_length):
            # account for each string of length pos (before EoS)
            self.nb_possible_sentences += (self.vocab_size)**(pos+1)
        
        sentenceId2sentence = np.zeros( (self.nb_possible_sentences, self.max_sentence_length))
        idx = 1
        local_token_pointer = 0
        global_token_pointer = 0
        while idx != self.nb_possible_sentences:
            sentenceId2sentence[idx] = sentenceId2sentence[idx-1]
            sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)

            while sentenceId2sentence[idx][local_token_pointer] == 0:
                # remove the possibility of an empty symbol on the left of actual tokens:
                sentenceId2sentence[idx][local_token_pointer] += 1
                local_token_pointer += 1
                sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)
            idx += 1
            local_token_pointer = 0    
        
        self.sentenceId2sentence = sentenceId2sentence

    def _make_infos(self, observations, infos):
        self.infos = []

        # Adapt info's legal_actions:
        for player_idx in range(self.nbr_agent):
            # Only No-op:
            legal_moves= [self.action_space.n-1]
            if player_idx==self.current_player:
                # Everything actually available, except No-op:
                # as int:
                available_directional_actions = copy.deepcopy(observations[player_idx]["available_directional_actions"])
                legal_moves = []
                for directional_action_idx in available_directional_actions[0]:
                    for sidx in range(self.nb_possible_sentences):
                        dir_action_sentence_action_idx = directional_action_idx*self.nb_possible_sentences + sidx
                        legal_moves.append(dir_action_sentence_action_idx)
            
            action_mask=np.zeros((1,self.action_space.n))
            np.put(action_mask, ind=legal_moves, v=1)
            
            info = copy.deepcopy(infos[player_idx])
            info['action_mask'] = action_mask
            info['legal_actions'] = action_mask
            self.infos.append(info)

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        
        self.nbr_agent = len(infos)
        self.current_player = infos[0]["current_player"].item()
        
        self._make_infos(observations, infos)

        return observations, copy.deepcopy(self.infos) 

    def _make_action(self, action):
        player_on_turn = self.infos[0]["current_player"].item()
        action = action[player_on_turn].item()

        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        if action<(self.action_space.n-1):
            original_action_direction_id = action // self.nb_possible_sentences
            
            original_action_sentence_id = (action % self.nb_possible_sentences)
            original_action_sentence = self.sentenceId2sentence[original_action_sentence_id:original_action_sentence_id+1] 
            # batch=1 x max_sentence_length
        else:
            # No-op action == skip directional + EoS message
            original_action_direction_id = 4
            original_action_sentence_id = 0
            original_action_sentence = self.sentenceId2sentence[original_action_sentence_id:original_action_sentence_id+1]
        
        ad = {
            'directional_action':original_action_direction_id,
            'communication_channel':original_action_sentence
        }
        
        return ad 
    
    def step(self, action):
        original_action = self._make_action(action)

        next_observations, reward, done, next_infos = self.env.step(original_action)

        self.nbr_agent = len(next_infos)
        self.current_player = next_infos[0]["current_player"].item()
        
        self._make_infos(next_observations, next_infos)

        return next_observations, reward, done, copy.deepcopy(self.infos)


class MultiBinaryCommunicationChannelWrapper(gym.Wrapper):
    """
    Assumes the :arg env: environment to have a Dict observation space,
    that contains the key "communication_channel", whose value is a MultiDiscrete.
    It transforms the MultiDiscrete observation in to a MultiBinary that is 
    the concatenation of each of the one-hot-encoded Discrete values.

    The communication channel allow for vocabulary_size ungrounded symbols
    and one grounded symbol that acts as EoS, whose index is 0.

    Args:
        env (gym.Env): Env to wrap. 
    """
    def __init__(self, env):
        super(MultiBinaryCommunicationChannelWrapper, self).__init__(env)

        self.observation_space = copy.deepcopy(env.observation_space)
        self.vocabulary_size = self.observation_space.spaces["communication_channel"].vocab_size
        self.max_sentence_length = self.observation_space.spaces["communication_channel"].max_sentence_length
        self.communication_channel_observation_space_size = self.max_sentence_length*(self.vocabulary_size+1)
        self.observation_space.spaces["communication_channel"] = gym.spaces.Discrete(self.communication_channel_observation_space_size)

        self.action_space = self.env.action_space 

    def _make_obs_infos(self, observations, infos):
        for player_idx in range(len(observations)):        
            token_start = 0
            new_communication_channel = np.zeros((1, self.communication_channel_observation_space_size))
            for token_idx in observations[player_idx]["communication_channel"]:
                new_communication_channel[0, int(token_start+token_idx)] = 1
                token_start += self.vocabulary_size+1
            observations[player_idx]["communication_channel"] = new_communication_channel
        return observations, infos

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        observations, infos = self._make_obs_infos(
            observations=observations,
            infos=infos,
        )
        return observations, infos

    def step(self, action):
        next_observations, reward, done, next_infos = self.env.step(action)
        next_observations, next_infos = self._make_obs_infos(
            observations=next_observations,
            infos=next_infos,
        )
        return next_observations, reward, done, next_infos



class ImgObservationWrapper(gym.Wrapper):
    """
    Assumes the :arg env: environment to have a Dict observation space,
    that contains the key 'image'.
    This wrapper makes the observation space consisting of solely the 'image' entry,
    while the other entries are put in the infos dictionnary.
    Args:
        env (gym.Env): Env to wrap.
    """

    def __init__(self, env):
        super(ImgObservationWrapper, self).__init__(env)
        
        self.observation_space = env.observation_space.spaces["image"]

        self.action_space = env.action_space 

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        nbr_agent = len(infos)
        
        new_observations = [obs["image"] for obs in observations]

        for agent_idx in range(nbr_agent):
            oobs = observations[agent_idx]

            for k,v in oobs.items():
                if k=="image":  continue
                infos[agent_idx][k] = v

        return new_observations, infos 
    
    def step(self, action):
        next_observations, reward, done, next_infos = self.env.step(action)        
        nbr_agent = len(next_infos)
        
        new_next_observations = [obs["image"] for obs in next_observations]

        for agent_idx in range(nbr_agent):
            oobs = next_observations[agent_idx]

            for k,v in oobs.items():
                if k=="image":  continue
                next_infos[agent_idx][k] = v
        
        return new_next_observations, reward, done, next_infos

    def render(self, mode='human', **kwargs):
        env = self.unwrapped
        return env.render(
            mode=mode,
            **kwargs,
        )
        

class OtherPlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super(OtherPlayWrapper, self).__init__(env)
        env = self.unwrapped
        self.nbr_players = env.nbr_players
        self.per_player_bijections = [
            Bijection(env=env)
            for _ in range(self.nbr_players)
        ]

    def reset(self, **kwargs):
        for pidx in range(self.nbr_players):
            self.per_player_bijections[pidx].reset()

        observations, infos = self.env.reset(**kwargs)

        for pidx in range(self.nbr_players):
            observations[pidx] = self.per_player_bijections[pidx].encode_obs(observations[pidx])

        return observations, infos

    def step(self, action, **kwargs):
        current_player = self.env.current_player
        new_action = self.per_player_bijections[self.current_player].decode_action(action)

        next_obs, reward, done, next_infos = self.env.step(new_action, **kwargs)

        for pidx in range(self.nbr_players):
            next_obs[pidx] = self.per_player_bijections[pidx].encode_obs(next_obs[pidx])

        return next_obs, reward, done, next_infos


def comaze_wrap(env, op=False):
    env = RGBImgWrapper(env)
    if op:
        print("Using Other-Play scheme!")
        env = OtherPlayWrapper(env)
    env = DiscreteCombinedActionWrapper(env)
    env = MultiBinaryCommunicationChannelWrapper(env)
    env = ImgObservationWrapper(env)
    return env