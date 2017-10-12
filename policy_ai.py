import pandas as pd
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
from frame import *
from random_ai import random_ai
from alpha_beta_ai import *

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
        clf = MLPClassifier(hidden_layer_sizes=(50,50),max_iter=2000,alpha=0.01)
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

if __name__ == '__main__':
    print("You're playing with policy_ai")
    interactive_play(policy_ai,[[0,0,0],[0,0,0],[0,0,0]])
