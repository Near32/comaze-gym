from typing import List, Dict

import torch
import torch.nn as nn

class ActionPolicy(nn.Module):
    def __init__(self, model:nn.Module):
        super(ActionPolicy, self).__init__()
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def clone(self, training=False):
        return ActionPolicy(model=self.model.clone(training=training))

    def forward(self, x:object):
        """
        :param x:
            Object representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the model.
        :return log_a:
            torch.Tensor of logits over actions 
            (as a Discrete OpenAI's action space).
        """
        raise NotImplementedError

    def reset(self, batch_size:int=None):
        """
        Call at the beginning of each episode.
        """
        raise NotImplementedError


from comaze_gym.utils import RuleBasedAgentWrapper

class RuleBasedActionPolicy(ActionPolicy):
    def __init__(
        self, 
        wrapped_rule_based_agent:RuleBasedAgentWrapper,
        combined_action_space:bool = False):
        """
        
        :param combined_action_space:
            If True, then the message and actions performed
            by the current agent are treated as belonging to
            the same OpenAI's Discrete action space of size 
            n= #messages * #actions.
            Else, n = # actions : directional actions.
        """
        super(RuleBasedActionPolicy, self).__init__(
            model=wrapped_rule_based_agent
        )
        self.combined_action_space = combined_action_space
    
    def clone(self, training=False):
        return RuleBasedActionPolicy(
            wrapped_rule_based_agent=self.model.clone(training=training), 
            combined_action_space=self.combined_action_space
        )
    
    def reset(self, batch_size:int):
        self.model.set_nbr_actor(batch_size)

    def get_nbr_actor(self):
        return self.model.get_nbr_actor()

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
        
        :return log_a:
            torch.Tensor of logits over actions 
            (as a Discrete OpenAI's action space).

            Here, depending on :attr combined_action_space:,
            we either marginalized over possible messages or not.
        """

        actions_idx = self.model.take_action(**x)
        # batch_size x 1

        batch_size = actions_idx.shape[0]
        self.action_space_dim = self.model.action_space_dim 
        
        # giving some entropy...
        p_a = torch.ones((batch_size, self.action_space_dim)) #.to(actions_idx.device)

        for bidx in range(batch_size):
            p_a[bidx, int(actions_idx[bidx])] = 2.0

        if self.combined_action_space:
            return p_a.log_softmax(dim=-1)

        # Otherwise, we sum over the messages dimension (excluding the NOOP action):
        self.vocab_size = (self.action_space_dim-1)//5
        # There are 5 possible directional actions:
        p_a = p_a[...,:-1].reshape((batch_size, 5, self.vocab_size)).sum(dim=-1)
        # batch_size x 5

        return p_a.log_softmax(dim=1)


        