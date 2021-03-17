import gym
import comaze_gym

skip = {'directional_action':4}
goleft = {'directional_action':0}
goright = {'directional_action':1}
goup = {'directional_action':2}
godown = {'directional_action':3}


def test_env_level1(reward_scheme):    
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-v0")
    
    # level : 1
    obs = env.reset()
    env.render()

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
    
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    env.render()
    import ipdb; ipdb.set_trace()

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
    obs = env.step(goleft)
    obs = env.step(goleft)

    obs = env.step(goleft)
    obs = env.step(goleft)
    
    env.render()
    import ipdb; ipdb.set_trace()
        
    obs = env.step(goleft)
    obs = env.step(goleft)

    env.render()
    import ipdb; ipdb.set_trace()

def test_env_fixed_actions(reward_scheme):    
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-FixedActions-v0")
    
    # level : 1
    obs = env.reset()
    env.render()

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


def test_env_level4_fixed_secrets(reward_scheme):    
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-SinglePlayerReward-Level4-FixedSecretGoalRules-v0")
    
    # level : 4
    obs = env.reset()
    env.render()

    import ipdb; ipdb.set_trace()

    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    obs = env.step(goleft)
    
    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    obs = env.step(godown)
    
    env.render()
    import ipdb; ipdb.set_trace()

    
    obs = env.step(godown)
    obs = env.step(godown)

    # retrieve time bonus:
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    env.render()
    import ipdb; ipdb.set_trace()    

    # go to blue:
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    obs = env.step(goup)
    obs = env.step(goup)
    
    env.render()
    import ipdb; ipdb.set_trace()    
    
    # go to yellow:
    obs = env.step(goright)
    obs = env.step(goright)
    
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
    obs = env.step(goup)
                

    obs = env.step(goleft)
    obs = env.step(goleft)
    
    
    env.render()
    import ipdb; ipdb.set_trace()
    
    env.close()



def test_env_single_player(reward_scheme):    
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-SinglePlayer-v0")
    
    # level : 1
    obs = env.reset()
    env.render()

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
    
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    env.render()
    import ipdb; ipdb.set_trace()

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
    obs = env.step(goleft)
    obs = env.step(goleft)

    obs = env.step(goleft)
    obs = env.step(goleft)
    
    env.render()
    import ipdb; ipdb.set_trace()
        
    obs = env.step(goleft)
    obs = env.step(goleft)

    env.render()
    import ipdb; ipdb.set_trace()

    #level 2:
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
    import ipdb; ipdb.set_trace()

    #level 3:
    env.render()

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
    import ipdb; ipdb.set_trace()
    obs = env.step(goup)
    import ipdb; ipdb.set_trace()
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
    import ipdb; ipdb.set_trace()
    
    
def test_env_level2(reward_scheme):    
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-v0")
    level = 2

    # level : 2
    obs = env.reset(level=level)
    env.render()

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
    import ipdb; ipdb.set_trace()
    
    env.close()    

def test_env_level3(reward_scheme):    
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-v0")
    level = 3

    # level : 3
    obs = env.reset(level=level)
    env.render()

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
    env.render()
    import ipdb; ipdb.set_trace()

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
    
    env.render()
    import ipdb; ipdb.set_trace()

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
    import ipdb; ipdb.set_trace()
    
    env.close()    

def test_env_level4(reward_scheme):    
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-v0")
    level = 4

    # level : 4
    obs = env.reset(level=level)
    env.render()

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
    import ipdb; ipdb.set_trace()
    
    env.close()


def test_env_level4_11x11(reward_scheme):    
    env = gym.make(f"CoMaze-11x11-{reward_scheme}-v0")
    level = 4

    # level : 4
    obs = env.reset(level=level)
    env.render()

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
    import ipdb; ipdb.set_trace()
    
    env.close()


def test_env_grey_goal(reward_scheme):    
    env = gym.make(f"CoMaze-7x7-{reward_scheme}-v0")
    
    # level : 1
    obs = env.reset()
    env.render()

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
    
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    obs = env.step(goright)
    
    env.render()
    import ipdb; ipdb.set_trace()

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
    obs = env.step(goleft)
    obs = env.step(goleft)

    obs = env.step(goleft)
    obs = env.step(goleft)
    
    env.render()
    import ipdb; ipdb.set_trace()
        
    obs = env.step(goleft)
    obs = env.step(goleft)

    env.render()
    import ipdb; ipdb.set_trace()
    

if __name__ == "__main__":
    #reward_scheme = "Sparse"
    #test_env_level1(reward_scheme)
    reward_scheme = "Dense"
    #test_env_single_player(reward_scheme)
    #test_env_fixed_actions(reward_scheme)


    test_env_level4_fixed_secrets(reward_scheme)

    #test_env_level1(reward_scheme)
    #test_env_level2(reward_scheme)
    #test_env_level3(reward_scheme)
    #test_env_level4(reward_scheme)

    #test_env_grey_goal(reward_scheme)

    test_env_level4_11x11(reward_scheme)