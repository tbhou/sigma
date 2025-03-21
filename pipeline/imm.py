from abc import abstractmethod

import math
import numpy as np
import torch


class Scheduler():
    """Noise scheduler."""

    def __init__(self, T=1.0, eps=0.0, sigma_d = 0.5):
        self.T = T
        self.eps = eps
        self.sigma_d = sigma_d

    @abstractmethod
    def alpha(self, t):
        pass

    @abstractmethod
    def sigma(self, t):
        pass

    def c_in(self, t):
        return 1.0 / math.sqrt(self.alpha(t) ** 2 + self.sigma(t) ** 2) / self.sigma_d
    
    def c_noise(self, t):
        return 1000 * t
    
    def snr(self, t):
        return (self.alpha(t) / self.sigma(t)) ** 2
    
    def logsnr(self, t):
        return math.log(self.snr(t))
    
    def sample_srt(self):
        t = np.random.uniform(self.eps, self.T)
        s = np.random.uniform(self.eps, t)
        r = max(s, t - 1.0 / (2 ** 10))
        return s, r, t
    
    def c_skip(self, s, t):
        return self.sigma(s) / self.sigma(t)
    
    def c_out(self, s, t):
        sigma_st = self.sigma(s) / self.sigma(t)
        alpha_s = self.alpha(s)
        alpha_t = self.alpha(t)
        return alpha_s - sigma_st * alpha_t


class ConsineScheduler(Scheduler):
    """VP-Consine scheduler."""

    def alpha(self, t):
        return math.cos(0.5 * math.pi * t)

    def sigma(self, t):
        return math.sin(0.5 * math.pi * t)
    
    def dt_logsnr(self, t):
        return -math.pi / (4 * self.alpha(t) * self.sigma(t))


class FMScheduler():
    """OT-FM scheduler."""

    def alpha(self, t):
        return 1 - t

    def sigma(self, t):
        return t
        
    def dt_logsnr(self, t):
        return -0.5 / (t * (1.0 - t))


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class IMM():
    """Inductive Moment Matching.
    
    An implementation of Inductive Moment Matching https://arxiv.org/abs/2503.07565.
    """

    def __init__(self, model, group_size, scheduler):
        self.model = model
        self.prev_model = model
        self.group_size = group_size
        self.scheduler = scheduler
        self.sigma_d = 0.25
    
    def ddim(self, x_t, x, s, t):
        return self.scheduler.c_out(s, t) * x + self.scheduler.c_skip(s, t) * x_t

    def f_theta(self, model, x_t, s, t):
        x_in = self.scheduler.c_in(t) * x_t
        s_in = self.scheduler.c_noise(s)
        t_in = self.scheduler.c_noise(t)
        x_out = model(x_in, s_in, t_in)
        return self.ddim(x_t, x_out, s, t)

    def kernel(self, x, y, s, t):
        d = (x - y).norm(2)
        eps = 1e-8
        d = torch.where(d < eps, eps, d)
        return torch.exp(-1.0 / self.scheduler.c_out(s, t) * d / x.shape[-1])
    
    def weight(self, s, t):
        b = 1.0  # Not specified in the paper.
        lambda_t = self.scheduler.logsnr(t)
        dt = -self.scheduler.dt_logsnr(t)
        a = self.scheduler.alpha(t) ** 2
        b = self.scheduler.sigma(t) ** 2
        return 0.5 * sigmoid(b - lambda_t) * dt * (a / a + b)

    def train_step(self, x):
        noise = torch.rand(*x.shape)
        num_groups = x.shape[0] // self.group_size
        x_groups = x.tensor_split(num_groups)
        noise_groups = noise.tensor_split(num_groups)
        loss = 0.0
        for x_group, noise_group in zip(x_groups, noise_groups):
            # for each group, sample s, r, t
            s, r, t = self.scheduler.sample_srt()
            
            # construct two training samples
            x_t = self.ddim(noise_group, x_group, t, 1)
            x_r = self.ddim(x_t, x_group, r, t)

            # run one-step denoising
            f_t = self.f_theta(self.model, x_t, s, t)
            with torch.no_grad():
                f_r_prev = self.f_theta(self.prev_model, x_r, s, r)
            
            # compute loss
            w = self.weight(s, t) / (self.group_size ** 2)
            local_loss = 0.0
            for j in range(self.group_size):
                for k in range(self.group_size):
                    k1 = self.kernel(f_t[j], f_t[k], s, t)
                    k2 = self.kernel(f_r_prev[j], f_r_prev[k], s, r)
                    k3 = self.kernel(f_t[j], f_r_prev[k], s, (t + r) * 0.5)
                    local_loss += k1 + k2 - 2 * k3
            loss += w * local_loss
        loss /= num_groups
        return loss

    def inference_step(self, N, x_shape):
        x = torch.rand(*x_shape)
        t = np.arange(0, N + 1) * 1.0 / N
        for i in range(N, 1, -1):
            x = self.f_theta(self.model, x, t[i-1], t[i])
        return x

    