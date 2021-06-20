from typing import List, Dict, Optional

import torch
import torch.nn as nn

class HiddenStatePolicy(nn.Module):
    def __init__(self, model:nn.Module):
        super(HiddenStatePolicy, self).__init__()
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def get_hidden_state_dim(self):
        return self.model.get_hidden_state_dim()

    def save_inner_state(self):
        raise NotImplementedError

    def restore_inner_state(self):
        raise NotImplementedError

    def clone(self, training=False):
        return HiddenStatePolicy(model=self.model.clone(training=training))

    def forward(self, x:object):
        """
        :param x:
            Object representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the model.
        :return hs:
            torch.Tensor of the hidden state of the model.
        """
        raise NotImplementedError

    #def reset(self, batch_size:int=None):
    def reset(self, batch_size:int, training:Optional[bool]=False):
        """
        Call at the beginning of each episode.
        """
        raise NotImplementedError


from comaze_gym.utils import RuleBasedAgentWrapper

class RuleBasedHiddenStatePolicy(HiddenStatePolicy):
    def __init__(
        self, 
        wrapped_rule_based_agent:RuleBasedAgentWrapper):
        """        
        """
        super(RuleBasedHiddenStatePolicy, self).__init__(
            model=wrapped_rule_based_agent
        )
    
    def get_hidden_state_dim(self):
        return torch.stack(
            [torch.from_numpy(hs).reshape(-1) for hs in self.model.get_hidden_state()],
            dim=0
        ).shape[-1]

    def clone(self, training=False):
        return RuleBasedHiddenStatePolicy(wrapped_rule_based_agent=self.model.clone(training=training))
    
    def reset(self, batch_size:int, training:Optional[bool]=False):
        self.model.set_nbr_actor(batch_size)

    def get_nbr_actor(self):
        return self.model.get_nbr_actor()

    def save_inner_state(self):
        self.saved_inner_state = self.model.get_rnn_states()

    def restore_inner_state(self):
        self.model.set_rnn_states(self.saved_inner_state)
        
    def forward(self, x:object):
        """
        :param x:
            Object representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the model.

            Here, x:Dict containing the keys:
            -'state': torch.Tensor containing the environment state.
            -'infos': Dict containing the entry 'abstract_repr' that is
                actually used by the :param model:RuleBasedAgentWrapper.
        
        :return hs:
            torch.Tensor of the hidden state of the model.
        """
        
        actions_idx = self.model.take_action(**x)
        # batch_size x 1

        hs = torch.stack(
            [torch.from_numpy(hs).reshape(-1) for hs in self.model.get_hidden_state()],
            dim=0
        ).float()
        # batch_size x hs_dim
        return hs


        