import os
from frame import *

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


if __name__ == '__main__':
    print("You're playing with alpha_beta_ai")
    interactive_play(alpha_beta_ai,[[0,0,0],[0,0,0],[0,0,0]])