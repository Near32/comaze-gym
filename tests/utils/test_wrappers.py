import gym
import comaze_gym
import numpy as np 


noop = [np.array([55])]*2


def test_wrapper(reward_scheme):    
    from comaze_gym.utils import comaze_wrap
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-v0")
    env = comaze_wrap(env)
    level = 4

    # level : 4
    obs = env.reset(level=level)
    env.render()

    obs = env.step(noop)

    """
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    # retrieve time bonus:
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(godown)
    obs = env.step(godown)
    
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goright)
    obs = env.step(goright)
    

    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    
    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goleft)
    obs = env.step(goleft)
    
    env.render()
    import ipdb; ipdb.set_trace()
    
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)

    # check that wall are not traversable:
    env.render()
    import ipdb; ipdb.set_trace()
    
    obs = env.step(goleft)
    obs = env.step(goleft)
    
    env.render()
    import ipdb; ipdb.set_trace()
    
    obs = env.step(godown)
    obs = env.step(godown)

    obs = env.step(goleft)
    obs = env.step(goleft)

    obs = env.step(goleft)
    obs = env.step(goleft)

    env.render()
    import ipdb; ipdb.set_trace()

    #entering last goal:
    obs = env.step(goup)
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goup)

    env.render()
    """
    import ipdb; ipdb.set_trace()
    
    env.close()


def test_env_level5(reward_scheme):    
    from comaze_gym.utils import comaze_wrap
    env = gym.make(f"CoMaze-9x9-{reward_scheme}-Level5-v0")
    env = comaze_wrap(env, op=True)
    
    obs = env.reset()
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    
    env.render()
    import ipdb; ipdb.set_trace()

    env.close()



def test_env_level5_easy_secrets(reward_scheme):    
    from comaze_gym.utils import comaze_wrap
    env = gym.make(f"CoMaze-9x9-{reward_scheme}-Level5-EasySecrets-v0")
    env = comaze_wrap(env, op=True)
    
    obs = env.reset()
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    
    env.render()
    import ipdb; ipdb.set_trace()

    env.close()


def test_env_level5_hard_secrets(reward_scheme):    
    from comaze_gym.utils import comaze_wrap
    env = gym.make(f"CoMaze-9x9-{reward_scheme}-Level5-HardSecrets-v0")
    env = comaze_wrap(env, op=True)
    
    obs = env.reset()
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    
    env.render()
    import ipdb; ipdb.set_trace()

    env.close()




def test_op_wrapper(reward_scheme):    
    from comaze_gym.utils import comaze_wrap
    #env = gym.make(f"CoMaze-7x7-{reward_scheme}-v0")
    env = gym.make(f"CoMaze-9x9-{reward_scheme}-Level5-HardSecrets-v0")
    env = comaze_wrap(env, op=True)
    #level = 1

    # level : 1
    #obs = env.reset(level=level)
    obs, info = env.reset()
    env.render()

    import ipdb; ipdb.set_trace()
    nobs, r, d, ninfo = env.step([np.array([55]), np.array([106])])
    #obs = env.step([np.array([55]), np.array([3])])
    import ipdb; ipdb.set_trace()
    
    
    obs = env.step(noop)

    """
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    # retrieve time bonus:
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(godown)
    obs = env.step(godown)
    
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goright)
    obs = env.step(goright)
    

    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    
    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goleft)
    obs = env.step(goleft)
    
    env.render()
    import ipdb; ipdb.set_trace()
    
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)

    # check that wall are not traversable:
    env.render()
    import ipdb; ipdb.set_trace()
    
    obs = env.step(goleft)
    obs = env.step(goleft)
    
    env.render()
    import ipdb; ipdb.set_trace()
    
    obs = env.step(godown)
    obs = env.step(godown)

    obs = env.step(goleft)
    obs = env.step(goleft)

    obs = env.step(goleft)
    obs = env.step(goleft)

    env.render()
    import ipdb; ipdb.set_trace()

    #entering last goal:
    obs = env.step(goup)
    env.render()
    import ipdb; ipdb.set_trace()

    obs = env.step(goup)

    env.render()
    """
    import ipdb; ipdb.set_trace()
    
    env.close()

if __name__ == "__main__":
    np.random.seed(1)
    reward_scheme = "Sparse"
    #test_wrapper(reward_scheme)
    reward_scheme = "Dense"
    #test_wrapper(reward_scheme)
    test_op_wrapper(reward_scheme)
    #test_env_level5(reward_scheme)
    #test_env_level5_easy_secrets(reward_scheme)
    #test_env_level5_hard_secrets(reward_scheme)

