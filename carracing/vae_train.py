'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import json
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae import ConvVAE, reset_graph

# Hyperparameters for ConvVAE
z_size=32
batch_size=20
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 1
DATA_DIR = "record"

def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset(filelist, N=1000, M=1000, returnactions=False): # N is 10000 episodes, M is number of timesteps
  data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
  actions = None
  idx = 0
  for i in range(N):
    filename = filelist[i]
    full_data = np.load(os.path.join("record", filename))
    raw_data = full_data['obs']
    action_info = full_data['action']
    if actions is None:
      actions = np.zeros((M*N, action_info.shape[1]), dtype=np.float32)
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      action_info = action_info[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    actions[idx:idx+l] = action_info
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  if returnactions:
    return data, actions
  else:
    return data

def save_json(data, jsonfile='vae.json'):
  with open(jsonfile, 'wt') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=0, separators=(',', ': '))

def vae_train():
  model_save_path = "tf_vae"
  if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

  # load dataset from record/*. only use first 10K, sorted by filename.
  filelist = os.listdir(DATA_DIR)
  filelist.sort()
  filelist = filelist[0:10000]
  length = len(filelist)
  global batch_size
  batch_size = min(batch_size, int(length / 2))
  #print("check total number of images:", count_length_of_filelist(filelist))
  dataset_full, actionlist = create_dataset(filelist, N=length, returnactions=True)

  # split into batches:
  total_length = len(dataset_full)
  num_batches = int(np.floor(total_length/batch_size))
  print("num_batches", num_batches)

  reset_graph()

  vae = ConvVAE(z_size=z_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                kl_tolerance=kl_tolerance,
                is_training=True,
                reuse=False,
                gpu_mode=True)

  # train loop:
  print("train", "step", "loss", "recon_loss", "kl_loss")
  for epoch in range(NUM_EPOCH):
    dataset = dataset_full[:]
    np.random.shuffle(dataset)
    for idx in range(num_batches):
      batch = dataset[idx*batch_size:(idx+1)*batch_size]

      obs = batch.astype(np.float)/255.0

      feed = {vae.x: obs,}

      (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
        vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
      ], feed)
    
      if ((train_step+1) % 500 == 0):
        print("step", (train_step+1), train_loss, r_loss, kl_loss)
      if ((train_step+1) % 5000 == 0):
        vae.save_json("tf_vae/vae.json")

  latent_full = np.zeros((len(dataset), vae.z_size), dtype=np.float32)
  for epoch in range(NUM_EPOCH):
    dataset = dataset_full
    for idx in range(num_batches):
      latent_full[idx*batch_size:(idx+1)*batch_size] = vae.encode(dataset[idx*batch_size:(idx+1)*batch_size])

  game_state = np.concatenate([latent_full, actionlist], 1)

  # finished, final model:
  save_json(game_state.tolist(),"game_data/game_states.json")
  vae.save_json("tf_vae/vae.json")

if __name__ == "__main__":
  vae_train()