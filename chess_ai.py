from stockfish import Stockfish
import chess
import numpy as np
import cv2
import model
import keras






# TODO:
# Chess Engine still needs testing but it seems viable so far?

# Play parameters
move_choice_dist = [0.75, 0.15, 0.1] # Probability distribution for choosing from the top_k moves.
is_white = True    # What color stockfish will be playing. True = white, False = black.
skill_level = 12
depth = 15
min_think_time = 15
elo = 1350
model = keras.models.load_model("occupancy_classifier.keras")

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
        #moves = stockfish.get_top_moves(len(move_choice_dist))
        #choose_randomly = True
        #for i in range(len(move_choice_dist)):
        #    move = moves[i]
        #if (choose_randomly): move = np.random.choice(moves, )
    else: # Player's turn

        img_captured_corners = None
        frame2 = None
        while True: # Change while loop condition later

            cam = cv2.VideoCapture(0)
            value, frame = cam.read()

            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            GRID = (7, 7)



            found, corners = cv2.findChessboardCorners(frame2, GRID, None)
            print("found: ", found, " corners: ", corners)
            cv2.imshow("Camera View", frame)

            
            if found:
                img_captured_corners = cv2.drawChessboardCorners(frame, GRID, corners, found)
                cv2.imshow("Camera View", img_captured_corners)
                break


            if cv2.waitKey(1) == ord('q'):
                break

        
        # Question! What is the purpose of gather_piece_data? #does this only work if you were given the metadata?
        # oh i see, when actually running this images should be the only thing it returns?
        images, piece_images, piece_labels, empty_labels = model.board_localization(image= frame2, corners= img_captured_corners, white_view= True, inner_grid= True, cw= 100, ch= 100, gather_piece_data= False ) # Assumes that it will always be white view

        str_labels = ["Empty", "Not Empty"]
        occupied_tiles = []
        all_occupancies = []
        for i in images:
      
            # Input img ---> img shape (100,100,3)
            img = np.expand_dims(img, 0) # ---> img shape now (1,100,100,3)
            # pred = model(img) ---> pred shape (1, 2)
            pred = np.reshape(pred, -1) # ---> pred shape now (2)
            # pred[0] = probability of 0th class, pred[1] = probability of 1st class
            # 0 class = empty, 1 class = occupied
            label = np.argmax(pred)
            all_occupancies.append(label)
            if (label == 1):
                occupied_tiles.append(label)
            
        
        str_labels = "PRNBQKprnbqk"
        all_pieces = []
        for i in occupied_tiles:
            # Input img ---> img shape (100,100,3)
            img = np.expand_dims(img, 0) # ---> img shape now (1,100,100,3)
            # pred = model(img) ---> pred shape (1, 12)
            pred = np.reshape(pred, -1) # ---> pred shape now (12)
            # pred[i] = probability of ith class
            # Classes are in order "PRNBQKprnbqk"
            all_pieces.append(label)


    

        # loop through the results and make a fenstring
        # then compare this fenstring with the previous fen string??
        # because we need to input the movement, not the board, i need to find out how to calculate the movment 



        move = input("What is your move (in algebraic notation)? ")
        if (stockfish.is_move_correct(move)): break
        print("Move invalid. Please try again.")


    # Update the board state.
    board.push(chess.Move.from_uci(move))
    current_move = not current_move
    move_number += 1