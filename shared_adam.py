# A3C için özelleştirilmiş Adam tabanlı optimizer

import math
import torch
import torch.optim as optim

# Amaç: Adam optimizerı paylaşımlı stateli şekilde tekrar tanımlamak
class SharedAdam(optim.Adam): # Adam optimizerdan türetilmiş SharedAdam nesnesi

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay) # Adam optimizerdan kalıtılanlar
        for group in self.param_groups: # self.param_groups optimizer için ağın ağırlıkları dahil tüm özellikleri bulundurur. 
            for p in group['params']: # Her ağırlık tensörü p'yi optimize etmek için

                state = self.state[p] # Başlangıçta, self.state boş bir dictionary nesnesidir. Yani, state = {} ve self.state = {p:{}} = {p: state} olmalıdır.
                
                state['step'] = torch.zeros(1) # state = {'step' : tensor([0])}
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_() # adam optimizer'ın güncellenmesi gradyanın üssel hareketli ortalamasına dayanır (moment 1)
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_() # adam optimizer'ın güncellenmesi ayrıca gradyanın karesinin üstel hareketli ortalamasına da dayanır (moment 2)

    # Hafıza (memory) paylaşımı
    # Buradaki amaç yapılan hesaplamaları tensor.cuda()'da olduğu gibi tüm threadler tarafından erişilebilir olmasını sağlamaktır. (parallellize thread)
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_() 
                state['exp_avg'].share_memory_() 
                state['exp_avg_sq'].share_memory_() 

    # Adam optimizasyon algoritmasının adım optimizasyonu fonksiyonu (https://arxiv.org/pdf/1412.6980.pdf buradakinin aynısı)
    # super(SharedAdam, self).step()'in aynısı
    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'][0]
                bias_correction2 = 1 - beta2 ** state['step'][0]
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss
