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
        ent_pi_m_x_it = []

        nbr_actors = self.message_policy.get_nbr_actor()

        if biasing:
            message_policy = self.message_policy
            self.message_policy.save_inner_state()
        else:
            message_policy = self.message_policy.clone()
        
        pi_bar_m = 0.0
        nbr_samples = 0
        nbr_eff_samples = 0

        for actor_id in range(batch_size):
            message_policy.reset(1)
        
            T = len(x[actor_id])
            if mask is None:
                eff_mask = torch.ones((batch_size, T))
            else:
                eff_mask = mask 
            
            nbr_eff_timesteps = 0
            for t in range(T):
                m = eff_mask[actor_id][t] 
                
                log_pt = message_policy(x[actor_id][t])
                pt = log_pt.exp()
                # 1 x message_space_dim
                
                m = m.to(pt.device)
                pi_bar_m += m*pt
                # 1 x message_space_dim
                nbr_samples += pt.shape[0]
                nbr_eff_samples += m*pt.shape[0]
                nbr_eff_timesteps += m

                ent_pt = -torch.sum(pt*log_pt, dim=-1, keepdim=True)
                # 1 x 1
                ent_pi_m_x_it.append(m*ent_pt)

                if self.target_ent_pt is None:
                    self.target_ent_pt = 0.5*torch.log(pt.shape[-1]*torch.ones(1)).to(pt.device).detach()

                L_ps_t = self.target_loss_lambda*torch.pow(ent_pt-self.target_ent_pt, 2.0)
                # 1
                
                if L_ps.device != L_ps_t.device:    L_ps = L_ps.to(L_ps_t.device)
                L_ps[actor_id:actor_id+1] += m*L_ps_t.reshape(-1)
                # batch_size
            
            # normalising:
            L_ps[actor_id:actor_id+1] /= nbr_eff_timesteps

        # normalization:
        pi_bar_m = pi_bar_m/nbr_eff_samples
        #pi_bar_m = pi_bar_m.softmax(dim=-1)
        log_pi_bar_m = pi_bar_m.log()
        # 1 x message_space_dim
        
        ent_pi_bar_m = -torch.sum(pi_bar_m*log_pi_bar_m, dim=-1, keepdim=True)
        # 1 x 1
        
        # Mutual information between agent's messages and trajectories:
        ent_pi_m_x_it = torch.cat(ent_pi_m_x_it, dim=-1)
        # (1 x nbr_samples)
        # WATCHOUT: nbr_samples is probably different from nbr_eff_samples... 
        # Thus, averaging requires division over nbr_eff_samples, not mean fn:
        exp_ent_pi_m_x_it_over_x_it = ent_pi_m_x_it.sum(dim=-1, keepdim=True)/nbr_eff_samples
        # (1 x 1 )

        # Loss is minimized when averaged policy's entropy is maximized:
        L_ps_ent_term = L_ps
        L_ps = L_ps-ent_pi_bar_m.reshape(-1)
        # batch_size 

        if biasing:
            self.message_policy.reset(nbr_actors, training=True)
            self.message_policy.restore_inner_state()
        
        rd = {
            "L_ps":L_ps,
            "L_ps_EntTerm": L_ps_ent_term,
            "ent_pi_bar_m": ent_pi_bar_m,
            "exp_ent_pi_m_x_it_over_x_it": exp_ent_pi_m_x_it_over_x_it,
        }
        
        return rd
