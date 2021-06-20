from typing import List, Dict 

import torch
import torch.nn.functional as F

import copy

from comaze_gym.metrics.message_policy import MessagePolicy


class MessageTrajectoryMutualInformationMetric(object):
    def __init__(self, message_policy:MessagePolicy):
        """
        
        :param message_policy:
            (Reference to an) MessagePolicy: Expects a (wrapped) torch.nn.Module that outputs logits
            over the possible messages (as a Discrete OpenAI's 
            action space).

        """
        self.message_policy = message_policy
        self.target_loss_lambda = 3.0 # cf paper appendix C.2
        self.target_ent_pt = None 

    def compute_pos_sign_loss(self, x:List[List[object]], mask:List[List[object]]=None, biasing:bool=False) -> torch.Tensor:
        """
        WARNING: this function resets the :attr message_policy:! 
        Beware of potentially erasing agent's current's internal states

        Notations refer to [Eccles et al, 2019] (https://arxiv.org/abs/1912.05676).
        :param x: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr message_policy:.

        :param mask:
            List[List[object]] containing, for each actor, at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid (e.g. episode not ended).
        """
        batch_size = len(x)
        
        L_ps = torch.zeros(batch_size)
        
        nbr_actors = self.message_policy.get_nbr_actor()

        if biasing:
            message_policy = self.message_policy
            self.message_policy.save_inner_state()
        else:
            message_policy = self.message_policy.clone()
        
        pi_bar_m = 0.0
        nbr_samples = 0

        for actor_id in range(batch_size):
            message_policy.reset(1)
        
            T = len(x[actor_id])
            if mask is None:
                eff_mask = torch.ones((batch_size, T))
            else:
                eff_mask = mask 

            for t in range(T):
                m = eff_mask[actor_id][t] 
                
                log_pt = message_policy(x[actor_id][t])
                pt = log_pt.exp()
                # 1 x message_space_dim
                
                m = m.to(pt.device)
                pi_bar_m += m*pt
                # 1 x message_space_dim
                nbr_samples += pt.shape[0]

                ent_pt = -sum([pt[...,i]*log_pt[...,i] for i in range(pt.shape[-1])])
                # 1 x 1
                
                if self.target_ent_pt is None:
                    self.target_ent_pt = 0.5*torch.log(pt.shape[-1]*torch.ones(1)).to(pt.device)

                L_ps_t = self.target_loss_lambda*torch.pow(ent_pt-self.target_ent_pt, 2.0)
                # 1
                
                if L_ps.device != L_ps_t.device:    L_ps = L_ps.to(L_ps_t.device)
                L_ps[actor_id:actor_id+1] += m*L_ps_t.reshape(-1)
                # batch_size

        # normalization:
        pi_bar_m = pi_bar_m/nbr_samples
        #pi_bar_m = pi_bar_m.softmax(dim=-1)
        log_pi_bar_m = pi_bar_m.log()
        # 1 x message_space_dim
        
        ent_pi_bar_m = -sum([pi_bar_m[...,i]*log_pi_bar_m[...,i] for i in range(pi_bar_m.shape[-1])])
        # 1 x 1
        
        # Loss is minimized when averaged entropy is maximized:
        L_ps = L_ps-nbr_samples*ent_pi_bar_m.reshape(-1)
        # batch_size 

        if biasing:
            self.message_policy.reset(nbr_actors, training=True)
            self.message_policy.restore_inner_state()

        return L_ps, ent_pi_bar_m