from frame import *

from alpha_beta_ai import alpha_beta_ai
from random_ai import random_ai
from policy_ai import policy_ai
from UCT_ai import UCT_ai 
from UCT_policy_ai import UCT_policy_ai
from q_learn_ai import q_learn_ai

if __name__ == '__main__':
    from frame import *
    ais = [alpha_beta_ai,policy_ai,UCT_ai,UCT_policy_ai,random_ai,q_learn_ai]
    for ai in ais:
        print("You're playing with {}".format(get_name(ai)))
        interactive_play(ai,[[0,0,0],[0,0,0],[0,0,0]])

