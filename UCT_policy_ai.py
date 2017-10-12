from UCT_ai import *
from policy_ai import *

def UCT_policy_ai(*args,**kwargs):
    return UCT_ai(*args,rollout_ai=policy_ai,**kwargs)

if __name__ == '__main__':
    from frame import interactive_play
    print("You're playing with UCT_policy_ai")
    interactive_play(UCT_policy_ai,[[0,0,0],[0,0,0],[0,0,0]])