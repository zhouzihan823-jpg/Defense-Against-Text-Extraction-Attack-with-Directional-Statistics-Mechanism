import math

import numpy as np
import torch
from scipy import integrate


from scipy.stats import vonmises_fisher
# The angular density f(x) = e^(-eps*x)*sin^(d-2)(x)
def f(x, d, eps): 
    return math.exp(-eps * x) * math.pow(math.sin(x), (d - 2))

def VMFMech(sentence_embeddings, eps):
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, dim=-1)
    n = sentence_embeddings.size(0)
    d = sentence_embeddings.size(1)
    kappa = 1.0 / (eps + 1e-8)
    sentence_embeddings_np = sentence_embeddings.cpu().detach().numpy()
    noisy_embeddings_np = np.zeros_like(sentence_embeddings_np)
    for i in range(n):
        mu = sentence_embeddings_np[i]
        sample = vonmises_fisher.rvs(mu, kappa, size=1)
        noisy_embeddings_np[i] = sample[0]

    noisy_embedd = torch.tensor(noisy_embeddings_np, 
                               device=sentence_embeddings.device,
                               dtype=sentence_embeddings.dtype)
    return noisy_embedd

# Exactly follow Alg.1 of CCS21 paper -- DP for Directional Data
def PurArc(d, eps):
    a = 0
    b = math.pi
    u = np.random.uniform(0, 1)
    for i in range(1, 25):
        theta = (a + b) / 2
        y = integrate.quad(f, 0, theta, args=(d, eps))[0] / integrate.quad(f, 0, math.pi, args=(d, eps))[0]
        if y < u:
            a = theta
        elif y > u:
            b = theta
    return theta
def PurMech(sentence_embeddings, eps):
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, dim=-1)
    n = sentence_embeddings.size(0)
    d = sentence_embeddings.size(1)
    direct = torch.randn((n, d), device=sentence_embeddings.device, requires_grad=False)
    direct_update = direct - torch.sum(direct * sentence_embeddings, dim=-1, keepdim=True) * sentence_embeddings
    direct_update = torch.nn.functional.normalize(direct_update, dim=-1)
    theta = torch.tensor([[PurArc(d, eps)] * d for _ in range(n)], dtype=torch.float, device=sentence_embeddings.device,
                         requires_grad=False)
    noisy_embedd = torch.cos(theta) * sentence_embeddings + torch.sin(theta) * direct_update
    return noisy_embedd
