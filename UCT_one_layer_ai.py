from copy import deepcopy
from UCT_ai import *

def UCT_one_layer(board,player,max_time=1000,rollout_ai=random_ai):
    root = Node(deepcopy(board),player,rollout_ai=rollout_ai)
    tic = 0
    toc = 0
    while toc - tic < max_time:
        node = root
        if node.status==0 and node.unvisited == [] and node.children != []:
            node = node.UCTSelectChild()
        if root.unvisited != [] and node.status==0:
            node = root.random_visit()
        score = node.rollout()
        while node != None:
            node.N += 1
            node.score += score
            node = node.parent
        toc += 1
    m = max(root.children,key=lambda c:c.N)
    m = m.last_move
    i = int(m/3)
    j = m - i*3
    return i,j

if __name__ == '__main__':
    from frame import interactive_play
    print("You're playing with UCT_one_layer_ai")
    interactive_play(UCT_one_layer_ai,[[0,0,0],[0,0,0],[0,0,0]])