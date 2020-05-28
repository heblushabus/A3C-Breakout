# A3C - Breakout

# Gerekli kütüphanelerin eklenmesi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bir tensörün değerlerini, belirli bir varyansa sahip olacak şekilde atayan fonksiyondur.
# Actor ve critic iki ayrı tamamen bağlı (fully-connected) katman olduğu için, bu fonksiyon sayesinde
# bu iki ayrı gruba farklı varyans değerlerinde atama yapabileceğiz.
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size()) # Önce rastgele normal dağılımlı değerler atıyoruz.
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # Bu işlem sonucunda, çıkışın varyansı standart sapmanın karesine eşit olacaktır. (var(out) = std^2)
    return out


# Sinir ağının farklı katmanları için farklı ağırlık atama işlemlerini yapan fonksiyondur.
# Amaç: Sinir ağının daha optimal bir öğrenme yapmasını sağlamak
# m: sinir ağı bağlantısı (katmanı)
def weights_init(m):
    classname = m.__class__.__name__ # katmanın tipini tutmak için (conv ya da fully-connected)

    if classname.find('Conv') != -1: # eğer bağlantı tipi convolutional ise
        weight_shape = list(m.weight.data.size()) # ağırlıkların boyutlarını tutan liste
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # ağırlık tensörünün büyüklüğü ile ters orantılı olarak uniform dağılımlı rastgele ağırlıklar üretir
        m.bias.data.fill_(0) # tüm bias değerlerini sıfırla

    elif classname.find('Linear') != -1: # eğer bağlantı tipi fully-connected ise
        weight_shape = list(m.weight.data.size()) # ağırlıkların boyutlarını tutan liste
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # ağırlık tensörünün büyüklüğü ile ters orantılı olarak uniform dağılımlı rastgele ağırlıklar üretir
        m.bias.data.fill_(0) # tüm bias değerlerini sıfırla


# A3C modelinin beyni
class ActorCritic(torch.nn.Module):

    # num_inputs: girdi olarak alınacak resmin boyutu
    # action_space: olası aksiyonların uzayı
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        # Actor ve Critic için ortak olarak kullanılan katmanlar:
        # Evrişimsel sinir ağı katmanları
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) # 3x3 boyutunda kernel uygulanarak 32 özellik dedektörlü ilk konvolüsyon katmanı oluşturulur.
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # Bir önceki katmandan aldığı 32 özelliği kullanarak yeni 32 özellik oluşturan ikinci konvolüsyon katmanı
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # Bir önceki katmandan aldığı 32 özelliği kullanarak yeni 32 özellik oluşturan üçüncü konvolüsyon katmanı
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # Bir önceki katmandan aldığı 32 özelliği kullanarak yeni 32 özellik oluşturan dördüncü konvolüsyon katmanı

        # Sinir ağımıza hafıza özelliği katan RNN katmanı
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256) # LSTM (Long Short Term Memory) - Oyunun bir olayını 256 boyutlu bir vectör uzayında kodlamış (encode) oluyoruz
        # LSTM in: 32*3*3
        # LSTM out: 256

        # Actor ve Critic için ayrı ayrı tanımlanan katmanlar:
        num_outputs = action_space.n # muhtemel aksiyonların sayısı

        self.critic_linear = nn.Linear(256, 1) # Critic için full connection: output = V(S)
        self.actor_linear = nn.Linear(256, num_outputs) # Actor için full connection: output = Q(S,A)
        self.apply(weights_init) # modelin ağırlıkları rastgele değerler ile tanımlanır
       
        # exploration vs exploitation
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) # Actor tensörü için ağırlıkların standart sapmasını 0.01 olarak ayarlıyoruz
        self.actor_linear.bias.data.fill_(0) # Actor'ün biasını sıfırlıyoruz (weights_init fonksiyonunda bunu zaten yapmıştık ama emin olmak için tekrar yapıyoruz)
        
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) # # Critic tensörü için ağırlıkların standart sapmasını 1.0 olarak ayarlıyoruz
        self.critic_linear.bias.data.fill_(0) # Critic'in biasını sıfırlıyoruz
       
        # LSTM katmanında iki tür bias vardır.
        self.lstm.bias_ih.data.fill_(0) # LSTM ih biasını sıfırlıyoruz
        self.lstm.bias_hh.data.fill_(0) # LSTM hh biasını sıfırlıyoruz
        self.train() # droupt-outs ve batch normalizationı aktive ederek train modunu ayarlıyor
     
    # İleri yayılım
    def forward(self, inputs):
        inputs, (hx, cx) = inputs # inputs değişkeni LSTM'in gizli katmanlarını da bulundurduğu için ayırıyoruz (hidden states, cell states)

        # Evirişimsel Sinir Ağı için ileri yayılım
        # eLU (Exponantial Linear Unit) kullanılma sebebi resim içerisindeki lineerliği kırıp, resim içerisindeki non-lineer ilişkileri ortaya çıkarmaktır.
        # ReLU: f(x) = max(0, x)
        # eLU: f(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
        x = F.elu(self.conv1(inputs)) # Giriş resminden aldığı sinyali 1. konvülasyon katmanına ileri yayılım yapar. Aktivasyon fonksiyonu olarak eLU kullanılıyor.
        x = F.elu(self.conv2(x)) # 1. Konvülasyon katmanından aldığı sinyali 2. konvülasyon katmanına ileri yayılım yapar.
        x = F.elu(self.conv3(x)) # 2. Konvülasyon katmanından aldığı sinyali 3. konvülasyon katmanına ileri yayılım yapar.
        x = F.elu(self.conv4(x)) # 3. Konvülasyon katmanından aldığı sinyali 4. konvülasyon katmanına ileri yayılım yapar.

        x = x.view(-1, 32 * 3 * 3) # Son konvülasyon katmanını 1D olan x vektörüne düzleştiriyoruz (flattening)
        
        hx, cx = self.lstm(x, (hx, cx)) # Düzleştirilen x ile eski hidden ve cell statesi kullanarak, LSTM'in yeni hidden ve cell statesi oluşturuyoruz
        
        x = hx # LSTM'in yapısı gereği çıktının anlamlı olması için x yerine hidden statesi kullıyoruz.
        
        # İki ayrı aktör(beyin) kullandığımız için, çıktı olarak iki ayrı sinyal üretiyoruz. (linear full connection)
        return self.critic_linear(x), self.actor_linear(x), (hx, cx) # Critic için V(S), Actor için Q(S,A), ve yeni hidden ve cell states (hx, cx)
