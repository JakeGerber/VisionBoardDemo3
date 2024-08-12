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

current_board_state = [0]*64

# squares = [
#             'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1',
#             'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2',
#             'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3',
#             'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4',
#             'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5',
#             'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6',
#             'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7',
#             'A8', 'B8', 'C8', 'D8', 'E8', 'F8', 'G8', 'H8',
#         ]

chess_squares = [
    chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1,
    chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2,
    chess.A3, chess.B3, chess.C3, chess.D3, chess.E3, chess.F3, chess.G3, chess.H3,
    chess.A4, chess.B4, chess.C4, chess.D4, chess.E4, chess.F4, chess.G4, chess.H4,
    chess.A5, chess.B5, chess.C5, chess.D5, chess.E5, chess.F5, chess.G5, chess.H5,
    chess.A6, chess.B6, chess.C6, chess.D6, chess.E6, chess.F6, chess.G6, chess.H6,
    chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7,
    chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8
]

print("Stockfish Setup Successful")
# print(stockfish.get_board_visual(not is_white))
print(board)


############### SCAN THE BOARD #######################
img_captured_corners = None # The corners returned by the chess board detector --> later fed into the board localization function
fullImage = None 
input("press the Enter key to scan board: ")
cam = cv2.VideoCapture(0)
while True: # Change while loop condition later

                
    value, frame = cam.read()
    fullImage = np.array(frame)
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    GRID = (7, 7)

    # Should i just save "corners" or "img_captured_corners"
    found, cornersList = cv2.findChessboardCorners(frame2, GRID, None)
    # print("found: ", found, " corners: ", cornersList)
    cv2.imshow("Camera View", frame)
    # print("joe biden 3")

    if found:
        img_captured_corners = cv2.drawChessboardCorners(frame, GRID, cornersList, found) # Can get rid of this in the final demo if we don't want to show the visualization
        cv2.imshow("Camera View", img_captured_corners)
        # print("FOUND CORNERS")
        break


    if cv2.waitKey(1) == ord('q'):
        break        


formattedCornersList = []
for i in cornersList:
    formattedCornersList.append([i[0][0], i[0][1]])
            













current_move = True # True = white, False = black
while True:
    
    # Display the current state of the game.
    stockfish.set_fen_position(board.fen())
    wdl_stats = stockfish.get_wdl_stats()
    if (wdl_stats != None): print("Predicted outcomes:\nWin: %.3f, Draw: %.3f, Lose: %.3f" % (wdl_stats[0]/1000, wdl_stats[1]/1000, wdl_stats[2]/1000))
    # print(stockfish.get_board_visual(not is_white))
    
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
        print("--------------AI's Turn--------------")
        print("Who's Turn: ")
        print(board.turn)
        move = stockfish.get_best_move()


        #moves = stockfish.get_top_moves(len(move_choice_dist))
        #choose_randomly = True
        #for i in range(len(move_choice_dist)):
        #    move = moves[i]
        #if (choose_randomly): move = np.random.choice(moves, )

        piece_iterator = 0

       

        board.push(chess.Move.from_uci(move))
        # chess.SQUARES
        
        print("joe biden")
        new_detected_board_state = []
        for square in chess_squares:
            # print(square)
            # square_coordinate = chess.SQUARES[square]
            # print(square_coordinate)
            #print(square_coordinate)

            print(board.piece_at(square))
            if board.piece_at(square) == None:
                new_detected_board_state.append("E")
            else:
                new_detected_board_state.append(str(board.piece_at(square)))
        print(new_detected_board_state)
        current_board_state = new_detected_board_state
        
        # for occ in all_occupancies:
        #     if occ == 0: # if empty
        #         new_detected_board_state.append("E")
        #     else: # if not empty
        #         new_detected_board_state.append(all_pieces[piece_iterator])
        #         piece_iterator += 1
        






    else: # Player's turn
        print("---------------Player's Turn-------------------")

        print("Who's Turn: ")
        print(board.turn)

        ############## STEP 1) CHESS BOARD CORNER DETECTION ##################
        # GOAL: Return the locations of the corners of every tile on the board & return the source image 


        # the black and white photo of the 

        # Continue to scan the video input (frame by frame) for a chessboard
        # If found, print visualization and break out of loop, also returns the corners of the chess board (only inner corners)
        # If not found, continue to scan for board
        # Note: Do we want to scan for the board every time it is the player's turn, or just once at the beginning 



        board_scan = False # Continue scanning until successfully scans the board
        fail_scan_coutner = 0
        while board_scan == False:

        # Add a space bar to start the scan
            input("press the Enter key to confirm your turn: ")


            cam = cv2.VideoCapture(0)
            while True: # Change while loop condition later
          
                value, frame = cam.read()
                fullImage = np.array(frame)
                frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                GRID = (7, 7)
                cv2.imshow("Camera View", frame)

                if cv2.waitKey(1) == ord('q'):
                    break        

            

            ############## STEP 2) CHESS BOARD LOCALIZATION / CUTTING UP IMAGE INTO 64 IMAGES OF EACH TILE ##################
            # INPUT: source image and inner grid corners
            # GOAL: Return a list of images of all 64 tiles


            # Question! What is the purpose of gather_piece_data? #does this only work if you were given the metadata?
            # oh i see, when actually running this images should be the only thing it returns?


            # Should only return 64 cut up images, the others should be empty arrays 

            # print(cornersList[0])
            # print(cornersList[0])
            # print("-----------------------")
            # print(cornersList[0][0])
            # print(len(cornersList[0][0]))

            
                # print(i[0])
                # print("gay")
                # print(i[0][0])
            # print(formattedCornersList)

            # only pass in corner pieces
            print("got here")
            images, piece_images, piece_labels, empty_labels = model.board_localization(image= fullImage, piece_data=[], corners= formattedCornersList
                                                                                        , white_view= True, inner_grid= True, cw= 100, ch= 100,
                                                                                          gather_piece_data= False ) # Assumes that it will always be white view
            
            for img in images:
                cv2.imshow("Tile View", img)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # input("enter")

            print("got here 2")
            
            ################################ RUN STEP 3 & 4 5 TIMES ###################################

            results_from_all_runs = []
            num_of_runs = 5

            for i in range(num_of_runs):

                ############## STEP 3) DETECTING TILE OCCUPANCIES ##################
                # INPUT: 64 cropped images of each tile
                # GOAL: Return a list of all tile's occupances, and a list of just the occupied tiles

                str_labels = ["Empty", "Not Empty"]
                occupied_tiles = []
                all_occupancies = []
                for img in images:
            
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

                    # print("all occupancies")
                    # print(all_occupancies)
                    # print("occupied tiles")
                    # print(occupied_tiles)



                ############## STEP 4) DETECTING TILE PIECES ##################
                # INPUT: The list of images of tiles that are occupied
                # GOAL: Return a list of all the tile's pieces
                
                str_labels = "PRNBQKprnbqk"
                all_pieces = []
                for i in range(64):
                    if all_occupancies[i] == 1:

                        # Input img ---> img shape (100,100,3)
                        img = np.expand_dims(images[i], 0) # ---> img shape now (1,100,100,3)
                        pred = piece_classifier_model(img) # ---> pred shape (1, 12)
                        pred = np.reshape(pred, -1) # ---> pred shape now (12)
                        # pred[i] = probability of ith class
                        # Classes are in order "PRNBQKprnbqk"
                        label = np.argmax(pred)
                        all_pieces.append(label)
                
                # print("all pieces")
                # print(all_pieces)



                # Combining the two arrays into one that states empty or piece type

                new_detected_board_state = []
                piece_iterator = 0
                str_labels = "PRNBQKprnbqk"

                for occ in all_occupancies:
                    if occ == 0: # if empty
                        new_detected_board_state.append("E")
                    else: # if not empty
                        new_detected_board_state.append(str_labels[all_pieces[piece_iterator]])
                        piece_iterator += 1
                print("board joey bidussy")
                print(new_detected_board_state)

                # ADD THE ALL PIECES RESULT TO THE results_from_5_runs List
                results_from_all_runs.append(new_detected_board_state)
            





            ################## STEP 4.5) TAKE THE MOST COMMON RESULT AND MAKE NEW_DETECTED_BOARD_STATE ########################
            # print("all runs results")
            # print(results_from_all_runs)
            final_new_detected_board_state = []
            for i in range(64):
                current_tile = []
                for j in range(num_of_runs):
                    current_tile.append(results_from_all_runs[j][i])
                final_new_detected_board_state.append(max(set(current_tile), key=current_tile.count))

            print("############## FINAL BOARD STATE ######################")
            print(final_new_detected_board_state)
            print("####################")
                




        
            ############## STEP 5) GVING RESULTS TO STOCKFISH ##################
            # INPUT: A list of all occupancies, and a list of all tiles with pieces classified
            # GOAL: Input the board data into stockfish and update board game state



                
                
            # print("new detected board state")
            # print(new_detected_board_state)
            # print(board.legal_moves)
            joe = list(board.legal_moves)
            # print(joe)
            # print("joe")
            # print(type(chess.Move.uci((joe[0]))))
            # print(chess.Move.uci((joe[0])))
            # # print(stockfish.get_board_visual(not is_white))
            # print(board)
            # board.push_uci(chess.Move.uci(joe[0]))
            # print("-------------------------------------")
            # # print(stockfish.get_board_visual(not is_white))
            # print(board)
            # # input("press the Enter key to continue: ")


#             [0, 4, 'E', 'E', 'E', 9, 'E', 7, 4, 3, 2, 'E', 'E', 'E', 'E', 8, 0, 'E', 4, 4, 'E', 'E', 9, 9, 'E', 0, 'E', 4, 0, 'E', 0, 9, 'E', 'E', 'E', 'E', 0, 3, 4, 5, 'E', 'E', 2, 2, 2, 4, 2, 0, 'E', 'E', 2, 2, 4, 'E', 2, 2, 
# 0, 0, 4, 2, 2, 'E', 'E', 4]

            str_labels = "PRNBQKprnbqk"
            old_board = [
                        7, 8, 9, 10, 11, 9, 8, 7,
                        6, 6, 6, 6, 6, 6, 6, 6,
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        0, 0, 0, 0, 0, 0, 0, 0,
                        1, 2, 3, 4, 5, 3, 2, 1
                        ]
            new_board = [
                        7, 8, 9, 10, 11, 9, 8, 7,
                        "E", 6, 6, 6, 6, 6, 6, 6,
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        6, "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        0, 0, 0, 0, 0, 0, 0, 0,
                        1, 2, 3, 4, 5, 3, 2, 1
                        ]
            
            current_board_state = old_board
            new_detected_board_state = new_board



        
            # then compare this fenstring with the previous fen string??
            # because we need to input the movement, not the board, i need to find out how to calculate the movment 


            print(new_detected_board_state)

            index_difference = []
            for i in range(64):
                if current_board_state[i] != new_detected_board_state[i]:
                    index_difference.append(i)

            print("index difference")
            print(index_difference)


            # if more than two differences, rescane (can there be more than movement per turn? can u hop like checkers? i forgor)

            # convert the index of the different tiles into the coordinates and write it in algebraic notation
            # maybe just hardcode this as a dictionary?

            # gives this movement to the stock fish api

            # board.piece_at() LOOK INTO THIS

            # Look at the two different tiles and append the black one first

            # board_location_dictionary = {1 : "h1", 2: "g1", 3: "f1", 4: "e1", 5: "d1", 6: "c1", 7: "b1", 8:"a1"}
            board_location_dictionary = {
                            0: "a8", 1: "b8", 2: "c8", 3: "d8", 4: "e8", 5: "f8", 6: "g8", 7: "h8",
                            8: "a7", 9: "b7", 10: "c7", 11: "d7", 12: "e7", 13: "f7", 14: "g7", 15: "h7",
                            16: "a6", 17: "b6", 18: "c6", 19: "d6", 20: "e6", 21: "f6", 22: "g6", 23: "h6",
                            24: "a5", 25: "b5", 26: "c5", 27: "d5", 28: "e5", 29: "f5", 30: "g5", 31: "h5",
                            32: "a4", 33: "b4", 34: "c4", 35: "d4", 36: "e4", 37: "f4", 38: "g4", 39: "h4",
                            40: "a3", 41: "b3", 42: "c3", 43: "d3", 44: "e3", 45: "f3", 46: "g3", 47: "h3",
                            48: "a2", 49: "b2", 50: "c2", 51: "d2", 52: "e2", 53: "f2", 54: "g2", 55: "h2",
                            56: "a1", 57: "b1", 58: "c1", 59: "d1", 60: "e1", 61: "f1", 62: "g1", 63: "h1"
                        }


            player_move = ""

            if len(index_difference) == 2:
                if new_detected_board_state[index_difference[0]] == "E":
                    player_move = player_move + board_location_dictionary[index_difference[0]]
                    player_move = player_move + board_location_dictionary[index_difference[1]]
                elif new_detected_board_state[index_difference[1]] == "E":
                    player_move = player_move + board_location_dictionary[index_difference[1]]
                    player_move = player_move + board_location_dictionary[index_difference[0]]
                else:
                    print("castling??")

            #detect castling
            if len(index_difference) == 4:
                if  new_detected_board_state[index_difference[62]] == "K" and new_detected_board_state[index_difference[61]] == "R":
                    player_move = "e1g1"
                elif new_detected_board_state[index_difference[58]] == "K" and new_detected_board_state[index_difference[61]] == "R":
                    player_move = "e1c1"
                elif new_detected_board_state[index_difference[6]] == "k" and new_detected_board_state[index_difference[5]] == "r":
                    player_move = "e8g8"
                elif new_detected_board_state[index_difference[2]] == "k" and new_detected_board_state[index_difference[3]] == "r":
                    player_move = "e8c8"

            
            # OKAY NOW ADD A FUNCTIONALITY FOR PROMOTION
           
            print("player move")
            print(player_move)
            # move = player_move



            #maybe use set_piece_at(), set_board_fen(), set_piece_map()
            

            ############## STEP 6) CORRECTING INCORRECT CHESS BOARD ##################
            # if (stockfish.is_move_correct(player_move)):
            # print(board)
            # print(list(board.legal_moves))
            if (chess.Move.from_uci(player_move) in board.legal_moves):
                board_scan = True
                board.push(chess.Move.from_uci(player_move))
                

                # print("------------------ GOT HERE ----------------------------------")
            else:
                fail_scan_coutner += 1
                print("----------------------- POOOOP ---------------------------------")
            if fail_scan_coutner >= 5:
                print("Move invalid. Please try again.")
                fail_scan_coutner = 0

        # board.push(chess.Move.from_uci(move))


        # print(board.legal_moves)
        # 





        

    # Update the board state.
    
    #print(stockfish.get_board_visual(not is_white))
    print(board)
    print("piece map")
    print(board.piece_map)
    #squares = chess.SquareSet()
    #print(squares)
    print(board.piece_at(chess.A1))
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