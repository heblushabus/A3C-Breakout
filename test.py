# Test Ajanı

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque

# Test Ajanı (Modeli güncellemiyor, sadece paylaşımlı modeli kullanarak ortamı keşfediyor.)
def test(rank, params, shared_model):
    torch.manual_seed(params.seed + rank) # Test ajanını asenkron yapmak için
    env = create_atari_env(params.env_name, video=True) # Ortamı video ile oynatmak için
    env.seed(params.seed + rank) # Ortamı asenkron yapmak için

    model = ActorCritic(env.observation_space.shape[0], env.action_space) # Modelin oluşturulması
    model.eval() # Modelin eğitim yapmaması için 

    state = env.reset() # input resmini numpy array olarak alıyoruz.
    state = torch.from_numpy(state) # Bunu torch tensörüne çeviriyoruz.
    reward_sum = 0
    done = True
    start_time = time.time() # Başlangıç zamanı
    actions = deque(maxlen=100) # https://pymotw.com/2/collections/deque.html
    episode_length = 0

    while True:
        episode_length += 1 # Bölüm uzunluğunu birer birer arttırıyoruz.
        if done: # Eğitim modundaki gibi paylaşımlı model ile senkronize hale getiriyoruz.
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)
        
        value, action_value, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(action_value)
        action = prob.max(1)[1].data.numpy() # Test Ajanı keşif yapmadan doğrudan en iyi aksiyonu kullanarak oynu oynar.
        state, reward, done, _ = env.step(action[0, 0]) # done = done or episode_length >= params.max_episode_length
        reward_sum += reward

        if done: # Her bölümün sonunda sonucu yazdırır.
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0 
            actions.clear()
            state = env.reset()
            time.sleep(60) # Öbür ajanları beklemek için 1 dk beklemesi için.
        state = torch.from_numpy(state) # Yeni durum (state) oluşturup devam eder.
