from typing import List, Dict 

import torch
import torch.nn.functional as F

import copy

from comaze_gym.metrics.action_policy import ActionPolicy


class MultiStepCIC(object):
    def __init__(self, action_policy:ActionPolicy, action_policy_bar:ActionPolicy):
        """
        
        :param action_policy:
            (Reference to an) ActionPolicy: Expects a (wrapped) torch.nn.Module that outputs logits
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
            lr=3e-4,
            #betas=[0.5,0.9],
        )

    def train_unconditioned_policy(self, x:List[List[object]], xp:List[List[object]], mask:List[List[object]]=None, a:List[object]=None) -> torch.Tensor:
        """
        Notations refer to [Eccles et al, 2019] (https://arxiv.org/abs/1912.05676).
        :param x: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr action_policy:.
        
        :param mask:
            List[List[object]] containing, for each actor, at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid (e.g. episode not ended).

        :param a: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the action of the current agent.
            If None, then argmax over :attr action_policy: is used.
            e.g.: the object can be a torch.Tensor containing
            the action indices (from a Discret OpenAI's action space)
            as outputed by the :attr action_policy:/current agent.

        :param xp: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent, with
            the messages being zeroed-out.
        """
        batch_size = len(x)

        use_argmax = (a == None)
        total_L_ce = torch.zeros(batch_size)
        
        action_policy = self.action_policy.clone()

        nbr_samples = 0
        accuracy = 0.0

        for actor_id in range(batch_size):
            L_ce = torch.zeros(1)
            
            action_policy.reset(1)
            self.action_policy_bar.reset(1)

            T = len(x[actor_id])
            if mask is None:
                eff_mask = torch.ones((batch_size, T))
            else:
                eff_mask = mask 

            for t in range(T):
                m = eff_mask[actor_id][t] 
                with torch.no_grad():
                    log_pt = action_policy(x[actor_id][t]).detach()
                    # batch_size x action_space_dim
                
                if use_argmax:
                    at = torch.argmax(log_pt, dim=-1)
                    # batch_size 
                else:
                    at = a[actor_id][t].detach()
                    # batch_size 
                
                log_p_bar_t = self.action_policy_bar(xp[actor_id][t])
                # batch_size x action_space_dim

                m = m.to(log_p_bar_t.device)
                at = at.to(log_p_bar_t.device)
                
                nbr_samples += m*at.shape[0]
                pred_at = log_p_bar_t.argmax(dim=-1)
                accuracy += m*(pred_at==at).float().sum()

                #L_ce_t = F.cross_entropy(
                L_ce_t = F.nll_loss(
                    input=log_p_bar_t,
                    target=at,
                    reduction='none'
                ).sum()#.sum(dim=-1)
                # 1
                
                #m = eff_mask[actor_id][t]
                #L_ce[actor_id] += m*L_ce_t
                if L_ce.device != L_ce_t.device:    L_ce = L_ce.to(L_ce_t.device)
                L_ce += m*L_ce_t
                # 1

            total_L_ce[actor_id] = L_ce

            self.optimizer.zero_grad()
            L_ce.mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        accuracy = accuracy/nbr_samples*100.0

        return total_L_ce, accuracy

    def compute_pos_lis_loss(self, x:List[List[object]], xp:List[List[object]], mask:List[List[object]]=None, biasing:bool=False) -> torch.Tensor:
        """
        WARNING: this function resets the :attr action_policy:! 
        Beware of potentially erasing agent's current's internal states

        Notations refer to [Eccles et al, 2019] (https://arxiv.org/abs/1912.05676).
        :param x: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr action_policy:.

        :param xp: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent, with
            the messages being zeroed-out.

        :param mask:
            List[List[object]] containing, for each actor, at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid (e.g. episode not ended).
        """
        batch_size = len(x)
        
        L_pl = torch.zeros(batch_size)
        
        nbr_actors = self.action_policy.get_nbr_actor()

        if biasing:
            action_policy = self.action_policy
            self.action_policy.save_inner_state()
        else:
            action_policy = self.action_policy.clone()
            
        for actor_id in range(batch_size):
            action_policy.reset(1)
            self.action_policy_bar.reset(1)
        
            T = len(x[actor_id])
            if mask is None:
                eff_mask = torch.ones((batch_size, T))
            else:
                eff_mask = mask 

            for t in range(T):
                m = eff_mask[actor_id][t] 
                
                pt = action_policy(x[actor_id][t]).exp()
                # 1 x action_space_dim
                p_bar_t = self.action_policy_bar(xp[actor_id][t]).exp().detach()
                # 1 x action_space_dim
                
                m = m.to(p_bar_t.device)
                pt = pt.to(p_bar_t.device)
                L_pl_t = -(p_bar_t-pt).abs().sum()#.sum(dim=-1)
                # 1
                
                #m = eff_mask[actor_id][t] 
                #L_pl[actor_id] += m*L_pl_t
                if L_pl.device != L_pl_t.device:    L_pl = L_pl.to(L_pl_t.device)
                L_pl[actor_id] += m*L_pl_t
                # batch_size

        if biasing:
            self.action_policy.reset(nbr_actors, training=True)
            self.action_policy.restore_inner_state()

        return L_pl

    def compute_multi_step_cic(self, x:List[List[object]], xp:List[List[object]], mask:List[List[object]]=None) -> torch.Tensor:
        """
        Notations refer to [Eccles et al, 2019] (https://arxiv.org/abs/1912.05676).
        :param x: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr action_policy:.

        :param xp: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent, with
            the messages being zeroed-out.
        
        :param mask:
            List[List[object]] containing, for each actor, at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid (e.g. episode not ended).
        """
        batch_size = len(x)
        
        ms_cic = torch.zeros(batch_size)
        
        action_policy = self.action_policy.clone()

        with torch.no_grad():
            for actor_id in range(batch_size):
                action_policy.reset(1)
                self.action_policy_bar.reset(1)
                
                T = len(x[actor_id])
                if mask is None:
                    eff_mask = torch.ones((batch_size, T))
                else:
                    eff_mask = mask 

                for t in range(T):
                    m = eff_mask[actor_id][t] 
                    
                    log_pt = action_policy(x[actor_id][t])
                    # 1 x action_space_dim
                    log_p_bar_t = self.action_policy_bar(xp[actor_id][t])
                    # 1 x action_space_dim
                    m = m.to(log_p_bar_t.device)
                    log_pt = log_pt.to(log_p_bar_t.device)
                    ms_cic_t = F.kl_div(
                        input=log_pt,
                        target=log_p_bar_t.exp(),
                        #log_target=True,
                        reduction='none',
                    ).sum()#.sum(dim=-1)
                    # 1 
                    
                    #m = eff_mask[actor_id][t]
                    #ms_cic[actor_id] += m*ms_cic_t
                    ms_cic[actor_id] += m*ms_cic_t
                    # 1

        return ms_cic