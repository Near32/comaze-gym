import gym
import comaze_gym

skip = 55
goleft = {'directional_action':0}
goright = {'directional_action':1}
goup = {'directional_action':2}
godown = {'directional_action':3}


def test_wrapper(reward_scheme):    
    from comaze_gym.utils import comaze_wrap
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-v0")
    env = comaze_wrap(env)
    level = 4

    # level : 4
    obs = env.reset(level=level)
    env.render()


    obs = env.step(skip)

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
    reward_scheme = "Sparse"
    #test_wrapper(reward_scheme)
    reward_scheme = "Dense"
    test_wrapper(reward_scheme)