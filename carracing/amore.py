#import concurrent.futures
import math 
import shutil
import os
import time
import argparse

from extract import experiment
from vae_train import vae_train
from series import serialize_rollouts
from rnn_train import rnn_train
import train



policy_list = None
minibatch_size = 20
INITIAL_REPS = 1
MAX_WORKERS = 1
DIR_NAME = "record"
ITERATIONS = 3
delete_data = False

def policy_train():

  parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))

  parser.add_argument('-o', '--optimizer', type=str, help='ses, pepg, openes, ga, cma.', default='cma')
  parser.add_argument('--num_episode', type=int, default=16, help='num episodes per trial')
  parser.add_argument('--eval_steps', type=int, default=25, help='evaluate every eval_steps step')
  parser.add_argument('-n', '--num_worker', type=int, default=2)
  parser.add_argument('-t', '--num_worker_trial', type=int, help='trials per worker', default=1)
  parser.add_argument('--antithetic', type=int, default=1, help='set to 0 to disable antithetic sampling')
  parser.add_argument('--cap_time', type=int, default=0, help='set to 0 to disable capping timesteps to 2x of average.')
  parser.add_argument('--retrain', type=int, default=0, help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
  parser.add_argument('-s', '--seed_start', type=int, default=0, help='initial seed')
  parser.add_argument('--sigma_init', type=float, default=0.1, help='sigma_init')
  parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')

  args = parser.parse_args()
  print(args.num_worker)
  if "parent" == train.mpi_fork(args.num_worker+1): os.exit()
  train.main(args)

if __name__ == '__main__':
  for i in range(ITERATIONS):
    print("Rolling out")
    # if policy_list is None:
    #     for i in range(INITIAL_REPS):
    #         experiment(MAX_TRIALS=2)
    # else: 
    #     policy_sublist = policy_list
    #     experiment(policy_sublist)
    
    # vae_train()
    # serialize_rollouts()
    # rnn_train()
    
    policy_list = policy_train()
    
  if delete_data: 
    shutil.rmtree(DIR_NAME)
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)