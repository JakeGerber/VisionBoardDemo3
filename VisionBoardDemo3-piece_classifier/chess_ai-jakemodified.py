from stockfish import Stockfish
import chess
import numpy as np

#from drawBoard import draw_checkerboard
#running into some issues I think it's easier to do this in the same python file
import tkinter as tk


def draw_triangle(canvas, x, y, size, color):
    # Different Triangle Vertices
    points = [
        x, y + size,  # Bottom-left
        x + size, y + size,  # Bottom-right
        x + size / 2, y  # Top
    ]
    canvas.create_polygon(points, fill=color, outline='black')



def draw_checkerboard(canvas, n, tileSize, boardArray):
    #TODO: useboard array to draw things to represent the pieces
    for i in range(n):
        for j in range(n):
            x, y = i * tileSize, j * tileSize
            if (i + j) % 2 == 0:
                canvas.create_rectangle(x, y, x + tileSize, y + tileSize, fill='white', outline='black')
                #draw_triangle(canvas, x, y, tileSize, 'black')
            else:
                canvas.create_rectangle(x, y, x + tileSize, y + tileSize, fill='black', outline='black')
                #draw_octagon(canvas, x, y, tileSize, 'white')
    

    canvas.create_oval(100, 100, 300, 300, outline="red", width=2)



#takes in board, converts to string, and sends to chess drawer python file.
def boardDraw(board, canvas):
    boardArray = [] #2D array of the board represented
    

    print("BOARD!:")
    print(str(board))

    row = []
    for x in str(board).split("\n"):
        for y in x:
            if (y == "."):
                row.append("Empty")
            elif (y != " "):
                addStr = ""
                if (y.isupper()):
                    addStr += "Player : "
                    y = y.lower()
                else:
                    addStr += "AI : "
                pieceDict = {"r" : "Rook", "n" : "Knight", "b" : "Bishop", "q" : "Queen", "k" : "King", "p" : "Pawn"}
                addStr += pieceDict[y]

                row.append(addStr)
        boardArray.append(row)
        row = []

    print(boardArray)

    canvas.update()
    n = 8
    tileSize = canvas_size // n
    draw_checkerboard(canvas, n, tileSize, boardArray)
    



# TODO:
# Chess Engine still needs testing but it seems viable so far?

def run_program(canvas):

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
    board = chess.Board()
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

        # Ask the player or stockfish for their move.
        if (current_move == is_white):  # Stockfish's turn (Currently W and Stockfish W or currently B and Stockfish B)
            move = stockfish.get_best_move()
        else: # Player's turn
            while True:
                print(board.board_fen())
                print(str(board)[0], " : ", print(type(board)))
                print(board.legal_moves)
                
                boardDraw(board, canvas) #probably not the right place to put the function

                move = input("What is your move (in algebraic notation)? ")
                if (stockfish.is_move_correct(move)): break
                print("Move invalid. Please try again.")
        # Update the board state.
        board.push(chess.Move.from_uci(move))
        current_move = not current_move
        move_number += 1


if __name__ == "__main__":
    print("WOWZA")
    root = tk.Tk()
    root.title("Checkerboard")

    canvas_size = 400
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size)
    canvas.pack()

    n = 8
    tileSize = canvas_size // n
    draw_checkerboard(canvas, n, tileSize)

    #root.mainloop() instead use tkinter_window.update()

    run_program(canvas)
