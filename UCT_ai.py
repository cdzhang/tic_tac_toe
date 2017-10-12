from random_ai import *
from copy import deepcopy
import random
import numpy as np
from frame import *
class Node:
    def __init__(self,board,player,rollout_ai=random_ai,last_move=-1):
        self.last_move = last_move
        self.board = deepcopy(board)
        self.player = player
        self.parent = None
        self.children = []
        self.unvisited = self.get_unvisited_actions()
        self.score = 0  #total score of player 0, whether self is player 0 or player 1
        self.N = 0
        self.rollout_ai = rollout_ai
        self.status = get_status(board)# -1 player 0 lose, 1 player 0 win, 2 tie, 0 not decided yet
    def get_unvisited_actions(self):
        unvisited_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    unvisited_actions.append(i*3+j)
        return unvisited_actions
    def UCTSelectChild(self):
        minmax = max if self.player == 0 else min
        return minmax(self.children,key=lambda c:c.score/c.N+np.sqrt(2*np.log(self.N)/c.N))
    def rollout(self):
        return tournament(self.rollout_ai,self.rollout_ai,deepcopy(self.board),self.player)
    def random_visit(self):
        m = random.choice(self.unvisited)
        self.unvisited.remove(m)
        i = int(m / 3)
        j = m - 3*i
        boardm = deepcopy(self.board)
        boardm[i][j] = 1 - 2*self.player
        random_child = Node(boardm,1-self.player,rollout_ai=self.rollout_ai,last_move = m)
        random_child.parent = self
        self.children.append(random_child)
        return random_child
    def print_board(self):
        print_state(self.board)
        print('____')
def UCT_ai(board,player,max_time=1000,rollout_ai=random_ai):
    #print_state(board)
    root = Node(deepcopy(board),player,rollout_ai=rollout_ai)
    tic = 0
    toc = 0
    while toc - tic < max_time:
        node = root
        while node.status==0 and node.unvisited == [] and node.children != []:
            node = node.UCTSelectChild()
        if node.unvisited != [] and node.status==0:
            node = node.random_visit()
        score = node.rollout()
        while node != None:
            node.N += 1
            node.score += score
            node = node.parent
        #toc = time.time()
        toc += 1
    m = max(root.children,key=lambda c:c.N)
    m = m.last_move
    i = int(m/3)
    j = m - i*3
    return i,j


def UCT_one_layer_ai(board,player,max_time=1000,rollout_ai=random_ai):
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
    from frame import *
    print("You're playing with UCT_ai")
    interactive_play(UCT_ai,[[0,0,0],[0,0,0],[0,0,0]])