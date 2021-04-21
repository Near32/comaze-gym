from typing import List, Dict

import torch
import torch.nn as nn

class ActionPolicy(nn.Module):
	def __init__(self, model:nn.Module):
		super(ActionPolicy, self).__init__()
		self.model = model

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

	def reset(self):
		"""
		Call at the beginning of each episode.
		"""
		raise NotImplementedError


from comaze_gym.utils import RuleBasedAgentWrapper

class RuleBasedActionPolicy(ActionPolicy):
	def __init__(
		self, 
		wrapped_rule_based_agent:RuleBasedAgentWrapper,
		combine_action_space:bool = False):
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

		batch_size = x["state"].shape[0]
		
		# This condition should only be called upon
		if self.model.get_nbr_actor() != batch_size:
			self.model.set_nbr_actor(batch_size)


		