from stockfish import Stockfish
import chess
import numpy as np

# TODO:
# Chess Engine still needs testing but it seems viable so far?

# Play parameters
move_choice_dist = [0.75, 0.15, 0.1] # Probability distribution for choosing from the top_k moves.
is_white = True    # What color stockfish will be playing. True = white, False = black.
skill_level = 12
depth = 15
min_think_time = 15
elo = 1350

# Stockfish & board objects
stockfish = Stockfish(path = "stockfish/stockfish-windows-x86-64-avx2.exe",
                      depth = depth, parameters = {
                          "Threads": 1,
                          "Minimum Thinking Time": min_think_time,
                          "Ponder": False,
                          # Universal Chess Interface, not University of California, Irvine :)
                          "UCI_Elo": elo
                      })
stockfish.set_skill_level(skill_level)
board = chess.Board("K6q/q6q/k7/8/8/8/8/8 w - -")
move_number = 1
print("Stockfish Setup Successful")

current_move = True # True = white, False = black
while True:
    # Display the current state of the game.
    stockfish.set_fen_position(board.fen())
    wdl_stats = stockfish.get_wdl_stats()
    if (wdl_stats != None): print("Predicted outcomes:\nWin: %.3f, Draw: %.3f, Lose: %.3f" % (wdl_stats[0]/1000, wdl_stats[1]/1000, wdl_stats[2]/1000))
    print(stockfish.get_board_visual(not is_white))
    
    # Check for win/draw/loss conditions.
    if (board.is_checkmate()):
        print(("Black" if current_move else "White") + " wins!")
        break
    if (board.is_stalemate()):
        print("Stalemate!")
        break
    # Check for check.
    if (board.is_check()): print("Check for " + ("White" if current_move else "Black") + "!")

    print(board.legal_moves)
    
    # Ask the player or stockfish for their move.
    if (current_move == is_white):  # Stockfish's turn (Currently W and Stockfish W or currently B and Stockfish B)
        move = stockfish.get_best_move()
        #moves = stockfish.get_top_moves(len(move_choice_dist))
        #choose_randomly = True
        #for i in range(len(move_choice_dist)):
        #    move = moves[i]
        #if (choose_randomly): move = np.random.choice(moves, )
    else: # Player's turn
        while True:
            move = input("What is your move (in algebraic notation)? ")
            if (stockfish.is_move_correct(move)): break
            print("Move invalid. Please try again.")
    # Update the board state.
    board.push(chess.Move.from_uci(move))
    current_move = not current_move
    move_number += 1