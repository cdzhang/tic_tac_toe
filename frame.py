import sys
write = sys.stdout.write
import random
import pandas as pd
import re

def get_status(board):
    mewin = [1,1,1]
    opwin = [-1,-1,-1]
    for i in range(3):
        if board[i] == mewin:
            return 1
        if board[i] == opwin:
            return -1
        vi = [s[i] for s in board]
        if vi == mewin:
            return 1
        if vi == opwin:
            return -1
    d1 = [board[0][0],board[1][1],board[2][2]]
    d2 = [board[0][2],board[1][1],board[2][0]]
    if d1 == mewin or d2 == mewin:
        return 1
    if d1 == opwin or d2 == opwin:
        return -1
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return 0
    return 2

def print_board(board):
    for i in range(3):
        for j in range(3):
            sij = board[i][j]
            if sij==0:
                write('. ')
            elif sij == 1:
                write('O ')
            else:
                write('X ')
        print('')
    print('______')


def tournament(ai0,ai1,start_board,start_player=0,print_out=False):
    board = start_board
    score = get_status(board)
    if score != 0:
        if score == 2:
            return 0
        return score
    player = start_player
    while score == 0:
        if print_out:
            print('player {} move'.format(player))
            print_board(board)
        i,j = ai0(board,player)
        board[i][j] = 1-2*player
        score = get_status(board)
        if score != 0:
            if print_out:
                print('player {} move'.format(player))
                print_board(board)
            if score == 2:
                return 0
            return score
        player = 1-player
        i,j = ai1(board,player)
        board[i][j] = 1-2*player
        score = get_status(board)
        if score != 0:
            if print_out:
                print('player {} move'.format(player))
                print_board(board)
            if score == 2:
                return 0
            return score
        player = 1-player

def compare_report(dic):
    vas = sorted([(key,value) for key,value in dic.items()])
    df = pd.DataFrame(vas)
    df.columns = ['outcome','count']
    df['outcome'] = ['lose','tie','win']
    N = sum(df['count'])
    def percent(row):
        row['percent'] = row['count']/N
        return row
    df=df.apply(percent,axis=1)
    df.set_index('outcome',inplace=True)
    return df
def print_report(df):
    N=sum(df['count'])
    def percent(f):
        return str(100*f)+'%'
    print('win:{}'.format(percent(df.loc['win']['percent'])))
    print('lose:{}'.format(percent(df.loc['lose']['percent'])))
    print('tie:{}'.format(percent(df.loc['tie']['percent'])))

def compare(ai1,ai2,times=100000):
    dic1={}
    dic2={}
    for i in range(times):
        board = [[0,0,0],[0,0,0],[0,0,0]]
        r1 = tournament(ai1,ai2,board)
        start_board = [[0,0,0],[0,0,0],[0,0,0]]
        r2 = tournament(ai2,ai1,[[0,0,0],[0,0,0],[0,0,0]])
        dic1[r1] = dic1.get(r1,0) + 1
        dic2[r2] = dic2.get(r2,0) + 1
        #if i % 100==0:print(i)
    for i in [-1,0,1]:
        dic1[i] = dic1.get(i,0)
        dic2[i] = dic2.get(i,0)
    df1 = compare_report(dic1)
    df2 = compare_report(dic2)
    print_report(df1)
    print('_______')
    print_report(df2)
    return dic1,dic2

def interactive_play(ai,board,start_player=0):
    st = 0
    rd = 1
    print_board(board)
    player = start_player
    if player == 1:
        i,j = ai(board,player)
        board[i][j] = -1
        print('computer move')
        print_board(board)
        if get_status(board) == -1:
            print('You Lose!')
            return
        elif get_status(board) == 2:
            print('Tie!')
            return
    while True:
        print('****rount {}:'.format(rd))
        ij = input('input i,j:\n')
        ij = ij.strip()
        ij = ij.split(',')
        i,j = int(ij[0]),int(ij[1])
        if board[i][j] != 0:
            print('illgegal!')
            continue
        board[i][j] = 1
        print('your move:')
        print_board(board)
        if get_status(board) == 1:
            print('You Win!')
            break
        elif get_status(board) == 2:
            print('Tie!')
            break
        i,j = ai(board,1-player)
        board[i][j] = -1
        print('computer move')
        print_board(board)
        if get_status(board) == -1:
            print('You Lose!')
            break
        elif get_status(board) == 2:
            print('Tie!')
            break
        rd += 1
    print('now exit')

def get_name(function):
    return str(function).split()[1]
if __name__ == '__main__':
    from ai import *
    ais = [alpha_beta_ai,policy_ai,UCT_ai,UCT_policy_ai,random_ai]
    for i in range(len(ais)):
        namei = get_name(ais[i])
        for j in range(i+1,len(ais)):
            namej = get_name(ais[j])
            print("compare {} and {}".format(namei,namej))
            if re.search('UCT',namei+namej):
                times = 1000
            else:
                times = 100000
            compare(ais[i],ais[j],times=times)
    #interactive_play(q_learn_ai,board)
