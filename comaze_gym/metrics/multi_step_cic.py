from typing import List, Dict 

import torch
import torch.nn.functional as F

import copy

from action_policy import ActionPolicy


class MultiStepCIC(object):
    def __init__(self, action_policy:ActionPolicy, action_policy_bar:ActionPolicy):
        """
        
        :param action_policy:
            ActionPolicy: Expects a (wrapped) torch.nn.Module that outputs logits
            over the possible actions (as a Discrete OpenAI's 
            action space).

        :param action_policy_bar:
            ActionPolicy: Expects a (wrapped) torch.nn.Module that outputs logits
            over the possible actions (as a Discrete OpenAI's 
            action space).
            If None, then a deepcopy of action_policy is made.

        """
        self.action_policy = action_policy
        if action_policy_bar is None: action_policy_bar = action_policy
        self.action_policy_bar = copy.deepcopy(action_policy_bar)
        self.optimizer = torch.optim.Adam(
            self.action_policy_bar.parameters(), 
            lr=1e-4,
            #betas=[0.5,0.9],
        )

    def train_unconditioned_policy(self, x:List[object], xp:List[object], mask:List[object]=None, a:List[object]=None):
        """
        Notations refer to [Eccles et al, 2019] (https://arxiv.org/abs/1912.05676).
        :param x: 
            List[object] containing at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr action_policy:.
        
        :param mask:
            List[object] containing at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid (e.g. episode not ended).

        :param a: 
            List[object] containing at each time step t an object
            representing the action of the current agent.
            If None, then argmax over :attr action_policy: is used.
            e.g.: the object can be a torch.Tensor containing
            the action indices (from a Discret OpenAI's action space)
            as outputed by the :attr action_policy:/current agent.

        :param xp: 
            List[object] containing at each time step t an object
            representing the observation of the current agent, with
            the messages being zeroed-out.
        """
        use_argmax = (a ==None)
        L_ce = None
        T = len(x)

        action_policy = copy.deepcopy(self.action_policy).reset()
        self.action_policy_bar.reset()

        for t in range(T):
            pt = action_policy(x[t]).detach()
            # batch_size x action_space_dim
            if use_argmax:
                at = torch.argmax(pt, dim=-1)
                # batch_size 
            else:
                at = a[t].detach()
                # batch_size 
            p_bar_t = self.action_policy_bar(xp[t])
            # batch_size x action_space_dim
            L_ce_t = F.cross_entropy(
                input=p_bar_t,
                target=at,
                reduction='none'
            ).sum(dim=-1)
            # batch_size
            import ipdb; ipdb.set_trace()
            if L_ce is None:    L_ce = torch.zeros_like(L_ce_t)
            
            if mask is None:
                m = torch.ones_like(L_pl_t)
            else:
                m = mask[t]

            L_ce += m*L_ce_t

        self.optimizer.zero_grad()
        L_ce.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def compute_pos_lis_loss(self, x:List[object], xp:List[object], mask:List[object]=None):
        """
        Notations refer to [Eccles et al, 2019] (https://arxiv.org/abs/1912.05676).
        :param x: 
            List[object] containing at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr action_policy:.

        :param xp: 
            List[object] containing at each time step t an object
            representing the observation of the current agent, with
            the messages being zeroed-out.

        :param mask:
            List[object] containing at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid (e.g. episode not ended).
        """
        L_pl = None
        T = len(x)

        self.action_policy.reset()
        self.action_policy_bar.reset()
        
        for t in range(T):
            pt = self.action_policy(x[t])
            # batch_size x action_space_dim
            p_bar_t = self.action_policy_bar(xp[t]).detach()
            # batch_size x action_space_dim
            L_pl_t = (p_bar_t-pt).abs().sum(dim=-1)
            # batch_size
            import ipdb; ipdb.set_trace()
            if L_pl is None:    L_pl = torch.zeros_like(L_pl_t)
            
            if mask is None:
                m = torch.ones_like(L_pl_t)
            else:
                m = mask[t] 
            
            L_pl += m*L_pl_t

        return L_pl

    def compute_multi_step_cic(self, x:List[object], xp:List[object], mask:List[object]=None):
        """
        Notations refer to [Eccles et al, 2019] (https://arxiv.org/abs/1912.05676).
        :param x: 
            List[object] containing at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr action_policy:.

        :param xp: 
            List[object] containing at each time step t an object
            representing the observation of the current agent, with
            the messages being zeroed-out.
        
        :param mask:
            List[object] containing at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid (e.g. episode not ended).
        """
        ms_cic = None
        T = len(x)

        action_policy = copy.deepcopy(self.action_policy).reset()
        self.action_policy_bar.reset()
        
        for t in range(T):
            pt = action_policy(x[t])
            # batch_size x action_space_dim
            p_bar_t = self.action_policy_bar(xp[t])
            # batch_size x action_space_dim
            ms_cic_t = F.kl_div(
                input=pt,
                target=p_bar_t,
                log_target=True,
                reduction=None,
            ).sum(dim=-1)
            import ipdb; ipdb.set_trace()
            # batch_size 
            if ms_cic is None:    ms_cic = torch.zeros_like(ms_cic_t)
            
            if mask is None:
                m = torch.ones_like(L_pl_t)
            else:
                m = mask[t]

            ms_cic += m*ms_cic_t

        return ms_cic