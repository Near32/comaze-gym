import copy
import numpy as np
import gym

from gym_minigrid.minigrid import *

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

    def _make_infos(self, observations):
        self.infos = []

        # Adapt info's legal_actions:
        for player_idx in range(self.nbr_agent):
            # Only No-op:
            legal_moves= [self.action_space.n-1]
            if player_idx==self.current_player:
                # Everything actually available, except No-op:
                # as int:
                available_directional_actions = observations[player_idx]["available_directional_actions"]
                legal_moves = []
                for directional_action_idx in available_directional_actions[0]:
                    for sidx in range(self.nb_possible_sentences):
                        dir_action_sentence_action_idx = directional_action_idx*self.nb_possible_sentences + sidx
                        legal_moves.append(dir_action_sentence_action_idx)
            
            action_mask=np.zeros((1,self.action_space.n))
            np.put(action_mask, ind=legal_moves, v=1)
            
            info = {}
            info['action_mask'] = action_mask
            info['legal_actions'] = action_mask
            self.infos.append(info)

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        
        self.nbr_agent = len(infos)
        self.current_player = infos[0]["current_player"]
        
        self._make_infos(observations)

        return observations, copy.deepcopy(self.infos) 

    def _make_action(self, action):
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
        self.current_player = next_infos[0]["current_player"]
        
        self._make_infos(next_observations)

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
        new_communication_channel = np.zeros((1, self.communication_channel_observation_space_size))
        
        for player_idx in range(len(observations)):        
            token_start = 0
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

    def setp(self, action):
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

def comaze_wrap(env):
    env = DiscreteCombinedActionWrapper(env)
    env = MultiBinaryCommunicationChannelWrapper(env)
    env = ImgObservationWrapper(env)
    return env