"""
Play a pre-trained model on a retro env
"""

import os
import sys
import retro
import datetime
import joblib
import argparse
import logging
import numpy as np
import pygame
from stable_baselines import logger

from common import init_env, init_model, init_play_env, get_model_file_name, GameDisplay

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--p1_alg', type=str, default='ppo2')
    parser.add_argument('--env', type=str, default='WWFArcade-Genesis')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--noserver', default=False, action='store_true')

    args = parser.parse_args(argv)

    logger.log("=========== Params ===========")
    logger.log(argv[1:])

    return args

def main(argv):
    
    logger.log('========= Init =============')
    args = parse_cmdline(argv[1:])

    play_env = init_play_env(args)
    p1_env = init_env(None, 1, args.state, 1, args)
    p1_model = init_model(None, args.load_p1_model, args.p1_alg, args, p1_env)

    display = GameDisplay(args) 

    logger.log('========= Start of Game Loop ==========')
    logger.log('Press ESC or Q to quit')
    state = play_env.reset()
    while True:
        framebuffer = play_env.render(mode='rgb_array')
        display.draw_frame(framebuffer)

        p1_actions = p1_model.predict(state)
            
        state, reward, done, info = play_env.step(p1_actions[0])

        if done:
            state = play_env.reset()


        keystate = display.GetInput()
        if keystate[pygame.K_q] or keystate[pygame.K_ESCAPE]:
            logger.log('Exiting...')
            break

if __name__ == '__main__':
    main(sys.argv)