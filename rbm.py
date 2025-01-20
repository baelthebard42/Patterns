import torch
import torch.nn as nn

class RestrictiveBM(nn.Module):

    def __init__(self, n_visible, n_hidden):
        super(RestrictiveBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.weights = torch.randn(n_hidden, n_visible)*0.01

        self.h_bias = torch.zeros(n_hidden)
        self.v_bias = torch.zeros(n_visible)

    
    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.weights.t()) + self.h_bias)
        return h_prob, torch.bernoulli(h_prob)
    
    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.weights) + self.v_bias)
        return v_prob, torch.bernoulli(v_prob)
    
    def reconstruction_error(self, v):
     v_prob, _ = self.sample_v(self.sample_h(v)[1])
     return torch.mean((v - v_prob) ** 2)
    

    def contrastive_divergence(self, v, lr, k=1):
        h_prob, h_sample = self.sample_h(v)
        for _ in range(k):
            v_prob, v_sample = self.sample_v(h_sample)
            h_prob, h_sample = self.sample_h(v_sample)

        positive_phase = torch.matmul(h_prob.t(), v)
        negative_phase = torch.matmul(h_sample.t(), v_sample)

        batch_size = v.size(0)
        

        self.weights += lr * (positive_phase - negative_phase) / batch_size
        self.v_bias += lr * torch.sum(v - v_sample, dim=0) / batch_size
        self.h_bias += lr * torch.sum(h_prob - h_sample, dim=0) / batch_size