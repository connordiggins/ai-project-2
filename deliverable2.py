#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:55:36 2020

@author: Brad Cooley

Sources:
 - Gym Source code: https://github.com/openai/gym
 - Getting started with gym: https://gym.openai.com/docs/
"""

# import statements
import time, gym, random, sys
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

def getTrainingData():
    
    # create a Ms Pacman environment
    env = gym.make('MsPacman-v0')
    env.reset()
    
    score_requirement = 0 #threshold for going into training model
    intial_games = 2 #how many games the training data comes from
    
    training_data = [] #data to pass into model
    accepted_scores = [] #scores recorded for training data
    
    for game_index in range(intial_games):
        score = 0
        game_memory = []
        previous_observation = []
        done = False #whether the game is over
        while not done:
                        
            # choose a random action
            action = random.randrange(8)
                    
            # get state data from the environment
            observation, reward, done, info = env.step(action)
                    
            # update the score
            score += reward
            
            # everything other than the first move
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = observation
            
            if done:
                break
        
        # update the training data if score is high enough
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0], action])
        
        env.reset()

    print(accepted_scores)
    
    # make the observations one dimensional
    for i in training_data:
        i[0] = i[0].flatten()
                
    return training_data

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

def train_model(training_data):
    X = np.array([i[0] for i in training_data]) # observation variable
    y = np.array([i[1] for i in training_data]) # action variable
    model = build_model(input_size=len(X[0]), output_size=len(y))
    
    model.fit(X, y, epochs=10)
    return model

def play_game(model):
    
    # create a Ms Pacman environment
    env = gym.make('MsPacman-v0')
    env.reset()
    
    scores = [] # people's scores
    games = 1 # number of games to play
    for each_game in range(games):
        score = 0
        done = False
        prev_obs = []
        while not done:
            #env.render()
            if len(prev_obs)==0:
                action = random.randrange(9)
            else:
                print(prev_obs)
                print(len(prev_obs))
                print(model.predict(prev_obs))
                action = model.predict(prev_obs)
            
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation.flatten()
            score+=reward
            if done:
                break
    
        env.reset()
        scores.append(score)

    print(scores)
    print('Average Score:', sum(scores)/len(scores))
        
training_data = getTrainingData()
model = train_model(training_data)
play_game(model)