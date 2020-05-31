# Başlangıç Noktası

from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
from envs import create_atari_env
from model import ActorCritic
from train import train
from test import test
import shared_adam

# Model parametreleri
class Params():
    def __init__(self):
        self.lr = 0.0001 # learning rate
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 4
        self.num_steps = 20
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0' # Sadece oynun adını değiştirerek diğer oyunlar üzerinde çalışabiliriz.


os.environ['OMP_NUM_THREADS'] = '1' # Her core için 1 thread
params = Params() # Varsayılan parametreler ile Params nesnesi oluşuturulur.
torch.manual_seed(params.seed) # Seed ayarı
env = create_atari_env(params.env_name) # Optimize edilmiş oyun ortamı
print(env.observation_space.shape)
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space) # Diğer ajanlar tarafından paylaşılan shared_model (Farklı corelardaki farklı threadlerde)
shared_model.share_memory() # Modeller farklı corelarda olsa bile paylaşımlı modeli kullanabilmesi için modeli paylaşımlı bellekte tutuyoruz.

optimizer = shared_adam.SharedAdam(shared_model.parameters(), lr=params.lr) 
optimizer.share_memory() # Paylaşımlı model üzerinde çalıştığı için bu da paylaşımlı bellekte tutulur.

processes = [] # process listesi
p = mp.Process(target=test, args=(params.num_processes, params, shared_model)) 
# allowing to create the 'test' process with some arguments 'args' passed to the 'test' target function - the 'test' process doesn't update the shared model but uses it on a part of it - torch.multiprocessing.Process runs a function in an independent thread
p.start() # p processini başlatır.
processes.append(p) # Başlayan processi process listesine ekler.

for rank in range(0, params.num_processes): # Paylaşımlı modeli güncellemesi için tüm processler eğitilir.
    p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
    p.start()
    processes.append(p)

for p in processes: # Programı güvenli durdurmak için
    p.join()
