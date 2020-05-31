# A3C modelinin eğitimi

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

# Tüm modellerin aynı gradyanı paylaştığından emin olmak için
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

# rank: Ajanları asenkron hale getirmek için seed'i kaydırmamızı sağlayacak parametre
# n tane ajan oluşacaksa, rank=n olacak ve her ajan için ortam, seed'i kaydırdığımız için, birbirinden bağımsız olacaktır.
# params: parametreler
# shared_model: ajanın küçük keşfini belirli sayıda adımda çalıştırmak için alacağı şeydir.
# optimizer: shared adam optimizer
def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank) # rank sayesinde seed'i sürekli kaydırarak ajanların asenkron olmasını sağlayacak
    # seed sayesinde aynı çevrimi tekrar tekrar yaptığımızda aynı sonucu alırız. Yani, model gelişimi boyunca deterministik bir ortam sağlar.
    env = create_atari_env(params.env_name) # Bu fonksiyon sayesinde optimize bir ortam oluşturuyoruz.
    env.seed(params.seed + rank) # Her ajan birbirinden bağımsız ortamlara sahiptir. Tüm bu ortamları belirli bir ortama hizalamak için env.seed() kullanılıyor. 

    model = ActorCritic(env.observation_space.shape[0], env.action_space) # ActorCritic sınıfına ait nesnenin oluşturulması
    state = env.reset() # input resmini 1*42*42 şeklinde 42x42'lik siyah-beyaz bir resim formatı olmasını sağlıyor (aslında bir numpy-array)
    state = torch.from_numpy(state) # numpy arrayi torch tensörüne çeviriyor

    done = True # Oyunun tamamlanıp tamamlanmadığını tutan değişken
    episode_length = 0 # Bölüm uzunluğunu tutan değişken

    while True:
        episode_length += 1 # bölüm uzunluğunu bir arttırıyoruz
        model.load_state_dict(shared_model.state_dict()) # Ajan, shared model ile num_steps kadar ortamda gözlem yapar.
        
        
        if done: # Eğer while döngüsünün ilk adımıysa ya da oyun bittiyse:
            cx = Variable(torch.zeros(1, 256)) # LSTM'in cell statesi sıfır olarak yeniden tanımlanıyor.
            hx = Variable(torch.zeros(1, 256)) # LSTM'in hidden statesi sıfır olarak yeniden tanımlanıyor.
        else: # Oyun bitmediyse:
            cx = Variable(cx.data) # Eski cell statesi tut.
            hx = Variable(hx.data) # Eski hidden statesi tut.

        # Öncüllemeler (initializations)
        values = [] # V(S) listesi (Critic'in çıktısı)
        log_probs = [] # log olasılıkları listesi
        rewards = [] # ödüller listesi
        entropies = [] # entropi listesi

        for step in range(params.num_steps): # num_steps kadar gözlem sayısı
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx))) 
            # Critic'in çıkışı V(S), Actor'ün çıkışı Q(S,A) ile yeni hidden ve cell states modelin çıktısı olarak üretilir. 
            # Yani, beyne bir giriş sinyali yolladık ve sonuç olarak bu değerler hesaplandı. 

            prob = F.softmax(action_values)
            # Softmax aktivasyon fonksiyonu ile Q değerlerinin olasılıksal dağılımlarını oluşturuyoruz. prob(a) = exp(prob(a))/sum_b(exp(prob(b)))
            # Entropi ile çalıştığımız için aynı zamanda logaritmik olasılığa da ihtiyacımız var.
            log_prob = F.log_softmax(action_values) 
            # Softmax aktivasyon fonksiyonu ile Q değerlerinin logaritmik olasılıksal dağılımlarını oluşturuyoruz. log_prob(a) = log(prob(a))
            

            entropy = -(log_prob * prob).sum(1) # H(p) = - sum_x p(x).log(p(x))
            entropies.append(entropy) # Hesaplanan entropiyi listede saklıyoruz.

            action = prob.multinomial(1).data # prob dağıtımından rastgele bir çizim yaparak bir eylem seçme
            log_prob = log_prob.gather(1, Variable(action)) # Seçilen aksiyonun logaritmik olasılığını alıyoruz.
            values.append(value) # Hesaplanan değer V(S) listeye ekleniyor.
            log_probs.append(log_prob) # Hesaplanan logaritmik olasılık listeye ekleniyor.

            # Seçilen aksiyon oyun ortamında gerçekleştirilir.
            state, reward, done, _ = env.step(action.numpy()) # Seçilen aksiyon oynanır, yeni duruma (state) geçilir ve yeni ödül kazanılır.
            done = (done or episode_length >= params.max_episode_length) # Eğer bölüm maksimum bölüm uzunluğundan uzun sürüyorsa oynu tamamlandı olarak işaretliyoruz. 
            # default max_episode_length: 10,000
            reward = max(min(reward, 1), -1) # Ödül -1 ile +1 arasında değerler alır.

            if done: # Eğer bölüm tamamlandıysa:
                episode_length = 0 
                state = env.reset() # Ortam sıfırlanır.

            state = torch.from_numpy(state) # Yeni durum oluşturulur.
            rewards.append(reward) # Beklenen yeni ödül listeye eklenir.

            if done:
                break # Gözlem durdurulur ve we doğrudan bir sonraki adıma geçilir. (Update shared_model)

            R = torch.zeros(1, 1) # Kümülatif ödül değeri tanımlanır.

            if not done: # Eğer bölüm tamamlanmadıysa:
                value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
                
                R = value.data # Son paylaşılan durumdaki değeri kümülatif ödül olarak tanımlarız.

            values.append(Variable(R)) # Son erişilen durumun değeri V(S) listeye eklenir.
            policy_loss = 0 # Actor
            value_loss = 0 # Critic
            R = Variable(R) # torch.Variable olduğuna emin olmak için
            gae = torch.zeros(1, 1) # Generalized Advantage Estimation tanımlanması: A(a,s) = Q(s,a) - V(s)

            for i in reversed(range(len(rewards))): # Son gözlem adımından başlanarak zamanda terse doğru gidilir.
                R = params.gamma * R + rewards[i] # Kümülatif ödül hesabı
                # R = gamma*R + r_t = 
                # R = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
                advantage = R - values[i] # R, zamanın Q'nun t = i anındaki bir tahmincisidir. Yani, advantage_i = Q_i - V(state_i) = R - value[i]
                value_loss = value_loss + 0.5 * advantage.pow(2) # Value loss hesaplanması

                TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data # Temporal Difference hesabı
                gae = gae * params.gamma * params.tau + TD 
                # gae = sum_i (gamma*tau)^i * TD(i) 
                # gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))

                policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] # Policy Loss hesabı
                # policy_loss = - sum_i log(pi_i)*gae + 0.01*H_i
                # pi: Aksiyonların olasılıksal dağılımlarının softmax aktivasyonu sonucu 
               
            # Stochastic Gradient Descent
            optimizer.zero_grad() # optimizer tanımlanması
            (policy_loss + 0.5 * value_loss).backward(retain_graph=True) # Policy Loss daha küçük bir değer olduğu için, Policy Loss'a Value Loss'tan 2 kat daha fazla önem veriyoruz.

            torch.nn.utils.clip_grad_norm(model.parameters(), 40) # Bu sayede gradyanların çok yüksek değerler alması önlenecektir. (Gradyanların normunun 0 ile 40 arasında olmasını sağlıyor.)
            ensure_shared_grads(model, shared_model) # Ajan ile paylaşımlı modelin aynı gradyanları kullandığından emin olmak için kullanıyoruz.
            optimizer.step() # Optimizasyon adımı çalıştırılır.
