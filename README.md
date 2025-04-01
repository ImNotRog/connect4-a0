# Connect 4 AlphaZero

This repo contains an implementation of AlphaZero for the game Connect4. It plays nearly perfectly (compared to previously known Connect-4 solvers), and you can play against it in the evaluate_model.py file.

The c4.py file is for training the model. The current model was trained on my M2 Macbook for around 5 hours. The self-play generation is optimized with the Python multiprocessing library.