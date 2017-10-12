import random
def random_ai(board,player):
    next_moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                next_moves.append((i,j))
    return random.choice(next_moves)

if __name__ == '__main__':
    from frame import *
    print("You're playing with random_ai")
    interactive_play(random_ai,[[0,0,0],[0,0,0],[0,0,0]])