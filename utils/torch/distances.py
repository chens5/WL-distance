# vim: ts=4 sw=4
"""Alternative torch implementation, based on sinkhorn instead of ot
the goal is to get a fast and differentiable loss
"""
import torch
from torch import Tensor
import torch.nn.functional as F
# from ot.bregman import sinkhorn_log as sinkhorn #recommended for computing gradients
# here we have to use OT / we can’t use Geomloss
# Because we need arbitrary cost matrices, as opposed to 
# having access to an embedding in R^n

def sinkhorn(a, b, C, epsilon, k=100, ):
    """batched version of sinkhorn"""

    *batch, n = a.shape
    *batch_, m = b.shape
    *batch__, n_, m_ = C.shape
    batch = torch.broadcast_shapes(batch, batch_, batch__)
    assert n == n_
    assert m == m_
    
    log_a = a.log()[..., :, None]
    log_b = b.log()[..., None, :]
    mC_eps = - C / epsilon

    f_eps = torch.randn((*batch, n, 1))#f over epsilon #batch, n
    g_eps = torch.randn((*batch, 1, m))#g over epsilon #batch, m
    for _ in range(k):
        f_eps = log_a - torch.logsumexp(mC_eps + g_eps, dim=-1, keepdim=True)
        g_eps = log_b - torch.logsumexp(mC_eps + f_eps, dim=-2, keepdim=True)
    log_P = mC_eps + f_eps + g_eps
    res = (C.log() + log_P).exp().sum((-1, -2))
    return res

def markov_measure(M: Tensor):
    """Takes a (batched) markov transition matrix, 
    and outputs its stationary distribution

    Args:
        M: the markov transition matrix
    """
    *b, n, n_ = M.shape
    assert n == n_
    target = torch.zeros((*b, n+1))
    target[..., n] = 1
    
    equations = (M.transpose(-1, -2) - torch.eye(n))
    equations = torch.cat([equations, torch.ones((*b, 1, n))], dim=-2)
    
    sol, *_ = torch.linalg.lstsq(equations, target)
    return sol.abs()
    


def wl_k(MX: Tensor, MY: Tensor, 
        l1: Tensor, l2: Tensor,
        k: int,
        muX: Tensor | None = None,
        muY: Tensor | None = None, 
        reg: float=.1, 
        sinkhorn_iter: int= 100
        ):
    """computes the WL distance

    computes the WL distance between two markov transition matrices 
    (represented as torch tensor)

    Batched over first dimension (b)

    Args:
        MX: (b, n, n) first transition tensor
        MY: (b, m, m) second transition tensor
        l1: (b, n,) label values for the first space
        l2: (b, m,) label values for the second space
        k: number of steps (k parameter for the WL distance)
        muX: stationary distribution for MX (if omitted, will be recomuputed)
        muY: stationary distribution for MY (if omitted, will be recomuputed)
        reg: regularization parameter for sinkhorn
        sinkhorn_iter: number of sinkhorn iterations for a step
    """
    b, n, n_ = MX.shape
    b_, m, m_ = MY.shape
    assert (n==n_) and (m == m_) and (b == b_)
    prev_matrix = (l1[:, :, None] - l2[:, None, :]).abs()
    cost_matrix = torch.zeros((n, m))
    

    for _ in range(k):
        # for i in range(n):
        #     cost_matrix[i] = sinkhorn(MX[i], MY.T, prev_matrix, reg=reg) #type:ignore
        cost_matrix = sinkhorn(
                MX[:, :, None, :], # b, n, 1, n
                MY[:, None, :, :], # b, 1, m, m
                prev_matrix[:, None, None, :, :], # b, 1, 1, n, m
                epsilon=reg, 
                k= sinkhorn_iter
        ) # b, n, m

    if muX is None: 
        muX = markov_measure(MX)
    if muY is None:
        muY = markov_measure(MY)

    return sinkhorn(muX, muY, cost_matrix, reg)
