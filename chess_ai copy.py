from stockfish import Stockfish
from os import listdir
import chess
import numpy as np
import cv2
import model
import keras
import pyautogui









# TODO:
# Chess Engine still needs testing but it seems viable so far?

# Play parameters
move_choice_dist = [0.75, 0.15, 0.1] # Probability distribution for choosing from the top_k moves.
is_white = True    # What color stockfish will be playing. True = white, False = black.
skill_level = 12
depth = 15
min_think_time = 15
elo = 1350
occupancy_classifier_model = keras.models.load_model("occupancy_classifier_finetuned.keras")
piece_classifier_model = keras.models.load_model("piece_classifier_finetuned.keras")



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

# chess_squares = [
#     chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1,
#     chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2,
#     chess.A3, chess.B3, chess.C3, chess.D3, chess.E3, chess.F3, chess.G3, chess.H3,
#     chess.A4, chess.B4, chess.C4, chess.D4, chess.E4, chess.F4, chess.G4, chess.H4,
#     chess.A5, chess.B5, chess.C5, chess.D5, chess.E5, chess.F5, chess.G5, chess.H5,
#     chess.A6, chess.B6, chess.C6, chess.D6, chess.E6, chess.F6, chess.G6, chess.H6,
#     chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7,
#     chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8
# ]

chess_squares = [
    chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8,
    chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7,
    chess.A6, chess.B6, chess.C6, chess.D6, chess.E6, chess.F6, chess.G6, chess.H6,
    chess.A5, chess.B5, chess.C5, chess.D5, chess.E5, chess.F5, chess.G5, chess.H5,
    chess.A4, chess.B4, chess.C4, chess.D4, chess.E4, chess.F4, chess.G4, chess.H4,
    chess.A3, chess.B3, chess.C3, chess.D3, chess.E3, chess.F3, chess.G3, chess.H3,
    chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2,
    chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1
]


print("Stockfish Setup Successful")
# print(stockfish.get_board_visual(not is_white))
print(board)
# for legal_board in board.legal_moves:
#     print(legal_board)




############### PRE-STEP 0) SCAN THE BOARD FOR THE CORNERS #######################
img_captured_corners = None # The corners returned by the chess board detector --> later fed into the board localization function
fullImage = None 
input("press the Enter key to scan board: ")
cam = cv2.VideoCapture(0)
while True: # Change while loop condition later
         
    value, frame = cam.read()
    fullImage = np.array(frame)
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    GRID = (7, 7)

    found, cornersList = cv2.findChessboardCorners(frame2, GRID, None)
    cv2.imshow("Camera View", frame)
    if found:
        img_captured_corners = cv2.drawChessboardCorners(frame, GRID, cornersList, found) # Can get rid of this in the final demo if we don't want to show the visualization
        cv2.imshow("Camera View", img_captured_corners)
        break
    if cv2.waitKey(1) == ord('q'):
        break    

# formats array to work with the board localization function
formattedCornersList = []
for i in cornersList:
    formattedCornersList.append([i[0][0], i[0][1]])


test_round = 0

input("Set up the board and press Enter to continue: ")
            

######################## MAIN GAME LOOP ##############################
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
    ################################## AI TURN ###########################################
    if (current_move == is_white):  # Stockfish's turn (Currently W and Stockfish W or currently B and Stockfish B)
        print("--------------AI's Turn--------------")
        move = stockfish.get_best_move()
        # if test_round == 0:
        #     move = "a2a4"
        #     test_round = 1
        print(list(board.legal_moves))
        #moves = stockfish.get_top_moves(len(move_choice_dist))
        #choose_randomly = True
        #for i in range(len(move_choice_dist)):
        #    move = moves[i]
        #if (choose_randomly): move = np.random.choice(moves, )
        piece_iterator = 0
        board.push(chess.Move.from_uci(move))

        # looks through the AIs turn and formats it to an array. This array is used to compare past and new states, which is important for understanding what the player's move was.
        new_detected_board_state = []
        for square in chess_squares:
            if board.piece_at(square) == None:
                new_detected_board_state.append("E")
            else:
                new_detected_board_state.append(str(board.piece_at(square)))
        # print(new_detected_board_state)
        current_board_state = new_detected_board_state
        
        

    ############################## PLAYERS TURN ########################################
    else: # Player's turn
        print("---------------Player's Turn-------------------")

        ############## STEP 1) CHESS BOARD FRAME SCAN ##################
        # GOAL: scan a frame of the board. This doesn't get any new coordinates, it uses the coordinates that were scanned at the begining
        # SO it is important that the board didn't move

        board_scan = False # Continue scanning until successfully scans the board
        fail_scan_coutner = 0

        while board_scan == False:

        # Add a space bar to start the scan
            input("press the Enter key to confirm your turn: ")
            print("during testing, PRESS Q to double confirm scan")
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


            print("Click on the picture and then click the right arrow to look at all the tile scans")
            # only pass in corner pieces # have not done that yet oops, but it still works!
            images, piece_images, piece_labels, empty_labels = model.board_localization(image= fullImage, piece_data=[], corners= formattedCornersList
                                                                                        , white_view= True, inner_grid= True, cw= 100, ch= 100,
                                                                                          gather_piece_data= False ) # Assumes that it will always be white view
            
            
            for img in images:
                cv2.imshow("Tile View", img)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # input("enter")

            
            ################################ RUN STEP 3 & 4 MULTIPLE TIMES ###################################

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
                # print("board joey bidussy")
                # print(new_detected_board_state)

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
            final_new_detected_board_state = results_from_all_runs[0]
            print("FINAL BOARD STATE")
            print(final_new_detected_board_state)
                






            ##################### NEW STEP 5) NEW BOARD ANALYSIS ###################################
            # print(list(board.legal_moves))

            # if (chess.Move.from_uci(player_move) in board.legal_moves):
            #     board_scan = True
            #     board.push(chess.Move.from_uci(player_move))

            str_labels = "PRNBQKprnbqk"

            old_board = [
                        "r", "n", "b", "q", "k", "b", "n", "r",
                        "p", "p", "p", "p", "p", "p", "p", "p",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "P", "P", "P", "P", "P", "P", "P", "P",
                        "R", "N", "B", "Q", "K", "B", "N", "R"
                        ]
            new_board = [
                        "r", "n", "b", "q", "k", "b", "n", "r",
                        "E", "p", "p", "p", "p", "p", "p", "p",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "p", "E", "E", "E", "E", "E", "E", "E",
                        "P", "E", "E", "E", "E", "E", "E", "E",
                        "E", "E", "E", "E", "E", "E", "E", "E",
                        "E", "P", "P", "P", "P", "P", "P", "P",
                        "R", "N", "B", "Q", "K", "B", "N", "R"
                        ]
            
            # final_new_detected_board_state = new_board




            # look through every legal move, then for each legal move make a new board, then compare that board to what was just scanned
            # if they are equal, stop the loop and push to the board
            # if none are the same, rescan the board 5 times
            # if still no legal move matches, tell the player that their move was illegal
            for legal_move in board.legal_moves:
                board_copy = board.copy()
                print(legal_move)
                print(type(legal_move))
                # print(board_copy)
                board_copy.push(legal_move)
                # print("possible legal board if move was taken")
                # print(board_copy)
                board_copy_state = []
                for square in chess_squares:
                    if board_copy.piece_at(square) == None:
                        board_copy_state.append("E")
                    else:
                        board_copy_state.append(str(board_copy.piece_at(square)))
                # board_copy_state.reverse()
                print("board copy state")
                #print(board_copy_state)
                for i in range(8):
                    for j in range(8):
                        print(board_copy_state[8*i + j], end="")
                    print()
                print("final new detected board state")
                for i in range(8):
                    for j in range(8):
                        print(final_new_detected_board_state[8*i + j], end="")
                    print()
                #print(final_new_detected_board_state)
                if board_copy_state == final_new_detected_board_state:
                    print(legal_move)
                    board.push(legal_move)
                    board_scan = True
                    fail_scan_coutner = 0
                    break

            
            if board_scan == False:
                fail_scan_coutner += 1
                print("----------------------- Please Rescan ---------------------------------")
            if fail_scan_coutner >= 5:
                print("-----------------Move invalid. Please try again.---------------------------------")
                fail_scan_coutner = 0

                




        
            # ############## STEP 5) GVING RESULTS TO STOCKFISH ##################
            # # INPUT: A list of all occupancies, and a list of all tiles with pieces classified
            # # GOAL: Input the board data into stockfish and update board game state


            # # These two board representations are fillers, until we can get the actualy board ai implemted
            # # NOTE: instead of numbers, it should be letters, but hopefully it will still work if that is changed... should be a small change

            # str_labels = "PRNBQKprnbqk"
            # old_board = [
            #             7, 8, 9, 10, 11, 9, 8, 7,
            #             6, 6, 6, 6, 6, 6, 6, 6,
            #             "E", "E", "E", "E", "E", "E", "E", "E",
            #             "E", "E", "E", "E", "E", "E", "E", "E",
            #             "E", "E", "E", "E", "E", "E", "E", "E",
            #             "E", "E", "E", "E", "E", "E", "E", "E",
            #             0, 0, 0, 0, 0, 0, 0, 0,
            #             1, 2, 3, 4, 5, 3, 2, 1
            #             ]
            # new_board = [
            #             7, 8, 9, 10, 11, 9, 8, 7,
            #             "E", 6, 6, 6, 6, 6, 6, 6,
            #             "E", "E", "E", "E", "E", "E", "E", "E",
            #             6, "E", "E", "E", "E", "E", "E", "E",
            #             "E", "E", "E", "E", "E", "E", "E", "E",
            #             "E", "E", "E", "E", "E", "E", "E", "E",
            #             0, 0, 0, 0, 0, 0, 0, 0,
            #             1, 2, 3, 4, 5, 3, 2, 1
            #             ]
            
            # current_board_state = old_board
            # new_detected_board_state = new_board
            # # print(new_detected_board_state)

            # # this goes through the old and new board state, and when it finds a difference, it stores the index of the tile
            # index_difference = []
            # for i in range(64):
            #     if current_board_state[i] != new_detected_board_state[i]:
            #         index_difference.append(i)

            # # print("index difference")
            # # print(index_difference)

            # board_location_dictionary = {
            #                 0: "a8", 1: "b8", 2: "c8", 3: "d8", 4: "e8", 5: "f8", 6: "g8", 7: "h8",
            #                 8: "a7", 9: "b7", 10: "c7", 11: "d7", 12: "e7", 13: "f7", 14: "g7", 15: "h7",
            #                 16: "a6", 17: "b6", 18: "c6", 19: "d6", 20: "e6", 21: "f6", 22: "g6", 23: "h6",
            #                 24: "a5", 25: "b5", 26: "c5", 27: "d5", 28: "e5", 29: "f5", 30: "g5", 31: "h5",
            #                 32: "a4", 33: "b4", 34: "c4", 35: "d4", 36: "e4", 37: "f4", 38: "g4", 39: "h4",
            #                 40: "a3", 41: "b3", 42: "c3", 43: "d3", 44: "e3", 45: "f3", 46: "g3", 47: "h3",
            #                 48: "a2", 49: "b2", 50: "c2", 51: "d2", 52: "e2", 53: "f2", 54: "g2", 55: "h2",
            #                 56: "a1", 57: "b1", 58: "c1", 59: "d1", 60: "e1", 61: "f1", 62: "g1", 63: "h1"
            #             }

            # player_move = ""


            # # looks through the index difference, and formats in the UCI move format. puts the empty tile first, then the tile where the piece moved
            # if len(index_difference) == 2:
            #     if new_detected_board_state[index_difference[0]] == "E":
            #         player_move = player_move + board_location_dictionary[index_difference[0]]
            #         player_move = player_move + board_location_dictionary[index_difference[1]]
            #     elif new_detected_board_state[index_difference[1]] == "E":
            #         player_move = player_move + board_location_dictionary[index_difference[1]]
            #         player_move = player_move + board_location_dictionary[index_difference[0]]
            #     else:
            #         print("castling??")

            # #detect castling. there are 4 castling cases, so i just hard coded them in
            # if len(index_difference) == 4:
            #     if  new_detected_board_state[index_difference[62]] == "K" and new_detected_board_state[index_difference[61]] == "R":
            #         player_move = "e1g1"
            #     elif new_detected_board_state[index_difference[58]] == "K" and new_detected_board_state[index_difference[61]] == "R":
            #         player_move = "e1c1"
            #     elif new_detected_board_state[index_difference[6]] == "k" and new_detected_board_state[index_difference[5]] == "r":
            #         player_move = "e8g8"
            #     elif new_detected_board_state[index_difference[2]] == "k" and new_detected_board_state[index_difference[3]] == "r":
            #         player_move = "e8c8"

            
            # # OKAY NOW ADD A FUNCTIONALITY FOR PROMOTION
           
            # print("player move")
            # print(player_move)
            # # move = player_move



            

            ############## STEP 6) CORRECTING INCORRECT CHESS BOARD ##################
            # if (stockfish.is_move_correct(player_move)):
            # print(board)
            # print(list(board.legal_moves))
            # if (chess.Move.from_uci(player_move) in board.legal_moves):
            #     board_scan = True
            #     board.push(chess.Move.from_uci(player_move))
                

            #     # print("------------------ GOT HERE ----------------------------------")
            # else:
            #     fail_scan_coutner += 1
            #     print("----------------------- Please Rescan ---------------------------------")
            # if fail_scan_coutner >= 5:
            #     print("Move invalid. Please try again.")
            #     fail_scan_coutner = 0


        

    # Update the board state.
    
    #print(stockfish.get_board_visual(not is_white))
    print(board)
    current_move = not current_move
    move_number += 1



