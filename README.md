# Breakout-A3C

Bu projede, [Asynchronous Advanced Actor Critic](https://arxiv.org/abs/1602.01783) algoritması kullanılarak [Breakout](https://en.wikipedia.org/wiki/Breakout_(video_game)) oynunu oynayan bir yapay zeka ajanı tasarlanmıştır.

## Proje Gereksinimleri
**Proje ortamı:** Ubuntu 18 <br>
* Bazı modüller Windows ortamında hata vermektedir.
  
**Dil:** Python 3.8.2

### Gereken Kütüphaneler
``` bash
pip install -r requirements.txt
```
### Olası Derleme Hataları
1. Atari ortamı ile ilgili bir hata alınırsa:
``` bash
pip install gym[atari]
```
2. Torch ile ilgili bir sorun yaşanırsa:
``` bash
conda install -c pytorch pytorch
```
3. OpenCV ile ilgili bir sorun yaşanırsa:
``` bash
conda install -c conda-forge opencv
```

## Nasıl çalıştırılır?
``` bash
python main.py
```
Yukarıdaki komut ile eğitim başlar. Varsayılan olarak her 10 iterasyonda bir dört ajanın oynadığı oynu mp4 video formatında `test` klasörüne kaydeder.
