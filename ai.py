from frame import *
import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy
import time
###   alpha_beta_ai #########
def load_dic_next(dic_file = 'dic_next.txt'):
    dic = {}
    if os.path.isfile(dic_file):
        with open(dic_file) as f:
            for line in f.readlines():
                line = line.strip()
                key,value = line.split(':')
                key = [int(k) for k in key.strip().split(',')]
                value = [int(k) for k in value.strip().split(',')]
                dic[tuple(key)] = tuple(value)
    return dic
def save_dic_next(dic,dic_file='dic_next.txt'):
    with open(dic_file,'w') as f:
        for key,value in dic.items():
            key = [str(k) for k in key]
            value = [str(v) for v in value]
            skey = ','.join(key)
            svalue = ','.join(value)
            f.write(skey+':'+svalue+'\n')

def hash_state(state,player):
    return tuple(state[0]) + tuple(state[1]) + tuple(state[2]) + (player,)

dic_next = load_dic_next()

def alpha_beta(cur_board,player):
    hs = hash_state(cur_board,player)
    if hs in dic_next:
        return dic_next[hs]
    st = get_status(cur_board)
    if st==2:
        return (0,None,None)
    if st!=0:
        return (st,None,None)
    minmax_score,minmax_i,minmax_j = -3+4*player,None,None
    pi = 1-2*player #player 0: 1, player 1:-1
    stack = []
    for i in range(3):
        for j in range(3):
            if cur_board[i][j] == 0:
                stack.append((i,j))
    random.shuffle(stack)
    for i,j in stack:
        cur_board[i][j] = pi
        if get_status(cur_board)==pi:
            cur_board[i][j] = 0
            dic_next[hs] = pi,i,j
            return pi,i,j
        step_score,next_i,next_j = alpha_beta(cur_board,1-player)
        cur_board[i][j] = 0
        if step_score == 2:
            step_score = 0
        if player == 0 and step_score >= minmax_score:
            minmax_score,minmax_i,minmax_j = step_score,i,j
        elif player == 1 and step_score <= minmax_score:
            minmax_score,minmax_i, minmax_j= step_score,i,j
    dic_next[hs] = minmax_score,minmax_i,minmax_j
    return minmax_score,minmax_i,minmax_j
def alpha_beta_ai(board,player):
    st,i,j = alpha_beta(board,player)
    return i,j

#save_dic_next(dic_next)
###############################

def random_ai(board,player):
    next_moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                next_moves.append((i,j))
    return random.choice(next_moves)
###########DL policy AI
def sample_state():
    '''
    get a random state through play of two random AIs.
    '''
    board=[[0,0,0],[0,0,0],[0,0,0]]
    player = random.randint(0,1)
    state_X = board[0] + board[1] + board[2] + [player]
    states = [tuple(state_X)]
    ai0 = random_ai
    ai1 = random_ai
    score = get_status(board)
    while score == 0:
        i,j = ai0(board,player)
        board[i][j] = 1-2*player
        score = get_status(board)
        if score != 0:
            break
        player = 1-player
        state_X = board[0] + board[1] + board[2] + [player]
        states.append(tuple(state_X))
        i,j = ai1(board,player)
        board[i][j] = 1-2*player
        score = get_status(board)
        if score != 0:
            break
        player = 1-player   
        state_X = board[0] + board[1] + board[2] + [player]
        states.append(tuple(state_X))
    return random.choice(states)

def gen_state_action_pair(board,player):
    '''
    get state-action pair, action is decided by alpha_beta_ai
    '''
    score, i,j = alpha_beta(board,player)
    return board[0] + board[1] + board[2] + [player] + [3*i + j]

def get_pairs_if_not_exist(pair_file='state_action_pairs.txt',times=100000):
    if not os.path.isfile(pair_file):
        print('generate state-action pairs')
        dic = {}
        with open(pair_file,'w') as f:
            for i in range(times):
                state = sample_state()
                if state not in dic:
                    b0 = list(state[:3])
                    b1 = list(state[3:6])
                    b2 = list(state[6:9])
                    player = state[9]
                    board = [b0,b1,b2]
                    sa = gen_state_action_pair(board,player)
                    sa = [str(sai) for sai in sa]
                    dic[state] = sa
                    f.write(','.join(sa)+'\n')
def get_clf(pair_file='state_action_pairs.txt',times=100000):
    get_pairs_if_not_exist()
    model_file = "MLPClassifier.sav"
    if not os.path.isfile(model_file):
        print('fitting model...')
        df = pd.read_csv('state_action_pairs.txt',names=['p0','p1','p2','p3','p4','p5','p6','p7','p8','player','k'])
        x = df[['p0','p1','p2','p3','p4','p5','p6','p7','p8','player']]
        y = df['k']
        clf = MLPClassifier(hidden_layer_sizes=(100,100),max_iter=2000,alpha=0.01)
        clf.fit(x,y)
        joblib.dump(clf,model_file)
    clf = joblib.load(model_file)
    return clf

clf = get_clf()


def cross_validation():
    df = pd.read_csv('state_action_pairs.txt',names=['p0','p1','p2','p3','p4','p5','p6','p7','p8','player','k'])
    x = df[['p0','p1','p2','p3','p4','p5','p6','p7','p8','player']]
    y = df['k']
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)
    layers = [2,3,4]
    sizes = [10,20,50,100]
    alphas = [0.0001,0.01,0.1]
    for layer in layers:
        for size in sizes:
            hidden_layer = tuple([size]*layer)
            for alpha in alphas:
                clf = clf = MLPClassifier(hidden_layer_sizes=hidden_layer,alpha=alpha,max_iter=2000)
                clf.fit(x_train,y_train)
                score_train = clf.score(x_train,y_train)
                score_test = clf.score(x_test,y_test)
                print('hidden_layers:{},alpha:{},score_train:{},score_test:{}'.format(hidden_layer,alpha,score_train,score_test))


def policy_ai(state,player,rpbs=False):
    state_X = state[0] + state[1] + state[2] + [player]
    state_X = np.array(state_X).reshape(1,-1)
    probilities = clf.predict_proba(state_X)
    probilities = probilities[0]
    pbs_next = sorted(list(zip(probilities,range(9))),reverse=True)
    if rpbs:
        return pbs_next
    while True:
        for prob,next_ij in pbs_next:
            i = int(next_ij / 3)
            j = next_ij - i*3
            if state[i][j] == 0:
                return i,j



#####UCT (MCTS + UCB)
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
def UCT(board,player,max_time=1000,rollout_ai=random_ai):
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
def UCT_policy_ai(*args,**kwargs):
    return UCT(*args,rollout_ai=policy_ai,**kwargs)

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


### Reinforcement AI
def save_Q_table(file='Q_table.txt'):
    with open(file,'w') as f:
        for key,value in Q_table.items():
            f.write('{}|{}\n'.format(key,value))
def load_Q_table(file='Q_table.txt'):
    dic = {}
    if os.path.isfile(file):
        with open(file) as f:
            for line in f.readlines():
                line = line.strip().split('|')
                dic[eval(line[0])] = qvalues(eval(line[1]))
    return dic

class qvalues:
    def __init__(self,*arg):
        self.qvs = {}
        if arg != None and len(arg)==2:
            a,v = arg
            self.update(a,v)
        if arg != None and len(arg)==1:
            for key,value in arg[0].items():
                self.update(key,value)
    def update(self,a,v):
        self.qvs[a] = v
    def max_av(self):
        ls = list(self.qvs.items())
        random.shuffle(ls)
        if len(ls) != 0:
            return max(ls,key=lambda x:x[1])
        else:
            return None
    def get_q(self,a):
        return self.qvs.get(a,0)
    def __str__(self):
        return str(self.qvs)

def retrieve_max_action_q(state):
    if state in Q_table:
        qv = Q_table[state]
        return qv.max_av()
    else:
        slots = [i for i in range(9) if state[i]==0]
        return random.choice(slots),0

def retrieve_q(state,action):
    if state in Q_table:
        return Q_table[state].get_q(action)
    else:
        return 0

def update_Q(state,action,value,printout=False):
    if state in Q_table:
        Q_table[state].update(action,value)
    else:
        Q_table[state] = qvalues(action,value)
    if printout:
        board = [list(state[:3]),list(state[3:6]),list(state[6:])]
        print_state(board)
        #print('action={},value={}'.format(action,value))

def get_legal_actions(board):
    actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                actions.append(i*3+j)
    return actions

def q_learn_ai(board,player):
    state = board[0] + board[1] + board[2]
    if player == 1:
        state = [-1*x for x in state]
    state = tuple(state)
    action,qs = retrieve_max_action_q(state)
    #assert(state[action]==0)
    if action == None:
        action = random.choice(get_legal_actions(board))
        #assert(state[action]==0)
    i = int(action / 3)
    j = action - 3*i
    #print(action)
    return i,j


def q_learn_episode(opponent_ai,alpha=0.5,epsilon=0.5,gamma=0.9):
    board = [[0,0,0],[0,0,0],[0,0,0]]
    player = random.randint(0,1)
    if player == 1:
        i,j = opponent_ai(board,player)
        board[i][j] = 1-2*player
        player = 1 - player
    while True:
        state = tuple(board[0] + board[1] + board[2])
        if np.random.uniform(0,1) <= epsilon:
            action = random.choice(get_legal_actions(board))
            i = int(action/3)
            j = action - 3*i
        else:
            i,j = q_learn_ai(board,0)
            action = 3*i+j
            #assert(state[action]==0)
        
        qs = retrieve_q(state,action)
        board[i][j] = 1
        score = get_status(board)
        if score != 0:#reward
            if score ==2:
                score = 0
            qs = (1-alpha)*qs + alpha*score
            #assert(state[action]==0)
            update_Q(state,action,qs)
            break
        i,j = opponent_ai(board,1)
        board[i][j] = -1
        score = get_status(board)
        if score != 0:
            if score == 2:
                score = 0
            qs = (1-alpha)*qs + alpha*score
            #assert(state[action]==0)
            update_Q(state,action,qs)
            break
        state2 = tuple(board[0] + board[1] + board[2])
        action2,qs2 = retrieve_max_action_q(state2)
        qs = (1-alpha)*qs + alpha*gamma*qs2
        #assert(state[action]==0)
        update_Q(state,action,qs)

def training(times=200000):
    for i in range(times):
        if i%10000==0:
            print(i)
        q_learn_episode(q_learn_ai)

def get_Q_table(Q_file='Q_table.txt',times=200000):
    if not os.path.isfile(Q_file):
        print('preparing Q table {}'.format(times))
        training(times)
        save_Q_table(Q_file)
Q_table = load_Q_table()
if not os.path.isfile('Q_table.txt'):
    get_Q_table()

if __name__ == '__main__':
    from frame import *
    ais = [alpha_beta_ai,policy_ai,UCT,UCT_policy_ai,random_ai,q_learn_ai]
    for ai in ais:
        print("You're playing with {}".format(get_name(ai)))
        interactive_play(ai,[[0,0,0],[0,0,0],[0,0,0]])

