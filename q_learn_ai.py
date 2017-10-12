from frame import *
import os
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
get_Q_table()
Q_table = load_Q_table()


if __name__ == '__main__':
    from frame import interactive_play
    print("You're playing with q_learn_ai")
    interactive_play(q_learn_ai,[[0,0,0],[0,0,0],[0,0,0]])

