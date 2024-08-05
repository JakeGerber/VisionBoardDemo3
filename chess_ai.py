from stockfish import Stockfish
from os import listdir
import chess
import numpy as np
import cv2
import model
import keras
import pyautogui


# I GOTTA COMMENT MY CODE FIRST



# IMPORT TEST DATA FILES
# train_files = listdir("/Users/derky/Desktop/VBDemo3/val")
# print(train_files)
# #train_files = listdir()
# #print(train_files)

# use_up_to = 2

# print("JOE BIDEN 1")
# print(train_files[0])
# im = None
# for imname in (train_files[:use_up_to] if use_up_to != None else train_files):
#             x = imname.split('.')
#             if (x[1] == "json"): continue
#             x = x[0]
#             print("JOE BIDEN")
#             print(x)
#             im = cv2.imread("/Users/derky/Desktop/VBDemo3/val/" + x + ".png")
#             print(type(im))

#             cv2.imshow("Image", im)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()


# imma just test each sectio individually


# TODO:
# Chess Engine still needs testing but it seems viable so far?

# Play parameters
move_choice_dist = [0.75, 0.15, 0.1] # Probability distribution for choosing from the top_k moves.
is_white = True    # What color stockfish will be playing. True = white, False = black.
skill_level = 12
depth = 15
min_think_time = 15
elo = 1350
occupancy_classifier_model = keras.models.load_model("occupancy_classifier.keras")
piece_classifier_model = keras.models.load_model("piece_classifier.keras")

current_board_state = []

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

        ############## STEP 1) CHESS BOARD CORNER DETECTION ##################
        # GOAL: Return the locations of the corners of every tile on the board & return the source image 


        img_captured_corners = None # The corners returned by the chess board detector --> later fed into the board localization function
        fullImage = None # the black and white photo of the 

        # Continue to scan the video input (frame by frame) for a chessboard
        # If found, print visualization and break out of loop, also returns the corners of the chess board (only inner corners)
        # If not found, continue to scan for board
        # Note: Do we want to scan for the board every time it is the player's turn, or just once at the beginning 

        board_scan = False # Continue scanning until successfully scans the board
        fail_scan_coutner = 0
        while board_scan == False:

        # Add a space bar to start the scan
            input("press the Enter key to continue: ")
            cam = cv2.VideoCapture(0)
        
            while True: # Change while loop condition later

                
                value, frame = cam.read()
                fullImage = frame
                # fullImage = im
                # frame = im
                frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                GRID = (7, 7)

                # Should i just save "corners" or "img_captured_corners"
                found, corners = cv2.findChessboardCorners(frame2, GRID, None)
                print("found: ", found, " corners: ", corners)
                cv2.imshow("Camera View", frame)
                print("joe biden 3")

                if found:
                    img_captured_corners = cv2.drawChessboardCorners(frame, GRID, corners, found) # Can get rid of this in the final demo if we don't want to show the visualization
                    cv2.imshow("Camera View", img_captured_corners)
                    print("FOUND CORNERS")
                    break
                

                if cv2.waitKey(1) == ord('q'):
                    break

            

            ############## STEP 2) CHESS BOARD LOCALIZATION / CUTTING UP IMAGE INTO 64 IMAGES OF EACH TILE ##################
            # INPUT: source image and inner grid corners
            # GOAL: Return a list of images of all 64 tiles


            # Question! What is the purpose of gather_piece_data? #does this only work if you were given the metadata?
            # oh i see, when actually running this images should be the only thing it returns?

            # Should only return 64 cut up images, the others should be empty arrays 
            images, piece_images, piece_labels, empty_labels = model.board_localization(image= fullImage, corners= img_captured_corners, white_view= True, inner_grid= True, cw= 100, ch= 100, gather_piece_data= False ) # Assumes that it will always be white view




            ############## STEP 3) DETECTING TILE OCCUPANCIES ##################
            # INPUT: 64 cropped images of each tile
            # GOAL: Return a list of all tile's occupances, and a list of just the occupied tiles

            str_labels = ["Empty", "Not Empty"]
            occupied_tiles = []
            all_occupancies = []
            for i in images:
        
                # Input img ---> img shape (100,100,3)
                img = np.expand_dims(img, 0) # ---> img shape now (1,100,100,3)
                pred = occupancy_classifier_model(img) # ---> pred shape (1, 2)
                pred = np.reshape(pred, -1) # ---> pred shape now (2)
                # pred[0] = probability of 0th class, pred[1] = probability of 1st class
                # 0 class = empty, 1 class = occupied
                label = np.argmax(pred)
                all_occupancies.append(label)
                if (label == 1):
                    occupied_tiles.append(label)



            ############## STEP 4) DETECTING TILE PIECES ##################
            # INPUT: The list of images of tiles that are occupied
            # GOAL: Return a list of all the tile's pieces
            
            str_labels = "PRNBQKprnbqk"
            all_pieces = []
            for i in occupied_tiles:
                # Input img ---> img shape (100,100,3)
                img = np.expand_dims(img, 0) # ---> img shape now (1,100,100,3)
                pred = piece_classifier_model(img) # ---> pred shape (1, 12)
                pred = np.reshape(pred, -1) # ---> pred shape now (12)
                # pred[i] = probability of ith class
                # Classes are in order "PRNBQKprnbqk"
                label = np.argmax(pred)
                all_pieces.append(label)


        
            ############## STEP 5) GVING RESULTS TO STOCKFISH ##################
            # INPUT: A list of all occupancies, and a list of all tiles with pieces classified
            # GOAL: Input the board data into stockfish and update board game state



            # Combining the two arrays into one that states empty or piece type

            new_detected_board_state = []
            piece_iterator = 0

            for occ in all_occupancies:
                if occ == 0: # if empty
                    new_detected_board_state.append("0")
                else: # if not empty
                    new_detected_board_state.append(all_pieces[piece_iterator])
                    piece_iterator += 1


        
            # then compare this fenstring with the previous fen string??
            # because we need to input the movement, not the board, i need to find out how to calculate the movment 

            index_difference = []
            for i in range(64):
                if current_board_state[i] != new_detected_board_state[i]:
                    index_difference.append(i)

            # if more than two differences, rescane (can there be more than movement per turn? can u hop like checkers? i forgor)

            # convert the index of the different tiles into the coordinates and write it in algebraic notation
            # maybe just hardcode this as a dictionary?

            # gives this movement to the stock fish api

            # board.piece_at() LOOK INTO THIS

            # Look at the two different tiles and append the black one first

            board_location_dictionary = {1 : "h1", 2: "g1", 3: "f1", 4: "e1", 5: "d1", 6: "c1", 7: "b1", 8:"a1"}
            player_move = ""

            if index_difference.len() == 2:
                if new_detected_board_state[index_difference[0]].islower():
                    player_move.append(board_location_dictionary[index_difference[0]])
                    player_move.append(board_location_dictionary[index_difference[1]])
                elif new_detected_board_state[index_difference[1]].islower():
                    player_move.append(board_location_dictionary[index_difference[1]])
                    player_move.append(board_location_dictionary[index_difference[0]])



            #maybe use set_piece_at(), set_board_fen(), set_piece_map()
            

            ############## STEP 6) CORRECTING INCORRECT CHESS BOARD ##################
            move = input("What is your move (in algebraic notation)? ")
            if (stockfish.is_move_correct(move)):
                board_scan = True
            else:
                fail_scan_coutner += 1
            if fail_scan_coutner >= 5:
                print("Move invalid. Please try again.")
                fail_scan_coutner = 0




        

    # Update the board state.
    board.push(chess.Move.from_uci(move))
    current_move = not current_move
    move_number += 1
















              # # loop through the results and make a fenstring
            # #   make a new list represnting the fenstring (or maybe string?)
            # #   loop through the list of all occupancies
            # #   if unoccupied, continue to go until you reach an occupied one, then add a number to the fenstring to represent the length of empty spaces to the list
            # #   if occupied, add the letter representing the piece to the list



            # # Turn the two array outputs from the model into a single fenstring
            # fenstring = ""
            # length_of_empty_spaces = 0
            # piece_iterator = 0
            # tile_counter = 0 # or 0?

            # for occ in all_occupancies:
            #     if occ == 0: # if empty
            #         length_of_empty_spaces += 1
            #         tile_counter += 1
            #     else: # if not empty
            #         if length_of_empty_spaces != 0:
            #             fenstring = fenstring + length_of_empty_spaces
            #             length_of_empty_spaces = 0
            #         fenstring = fenstring + all_pieces[piece_iterator]
            #         piece_iterator += 1
            #         tile_counter += 1
            #     if tile_counter % 8 == 0 and tile_counter != 64: # && 
            #         fenstring = fenstring + "/" # does this accidentally add a dash at the end of the fenstring