"""Hand-written numpy W(theta) and grad L for an MPS (core legs: site, left bond, right bond)."""
import numpy as np


class FastMPS:
    def __init__(self, net, w_star):
        assert net.kind == "mps"
        self.net, self.N, self.d = net, net.n, net.d
        self.ws = np.asarray(w_star).reshape(-1)

    def cores(self, th):
        n = self.net
        out = []
        for k in range(self.N):
            g = th[n.off[k] : n.off[k + 1]].reshape(n.shapes[k])
            if k == 0:
                g = g[:, None, :]  # (d, 1, chi)
            out.append(g)
        return out

    def lefts(self, G):
        A = [G[0][:, 0, :]]
        for k in range(1, self.N):
            g = G[k].transpose(1, 0, 2)  # (chi_l, d, chi_r)
            A.append((A[-1] @ g.reshape(g.shape[0], -1)).reshape(-1, g.shape[2]))
        return A

    def W(self, th):
        return self.lefts(self.cores(th))[-1][:, 0]

    def loss_grad(self, th):
        G = self.cores(th)
        A = self.lefts(G)
        E = A[-1][:, 0] - self.ws
        Bs = [None] * (self.N + 1)
        Bs[self.N] = np.ones((1, 1))
        for k in range(self.N - 1, 0, -1):
            g = G[k].transpose(1, 0, 2)  # (chi_l, d, chi_r)
            Bs[k] = (g.reshape(-1, g.shape[2]) @ Bs[k + 1]).reshape(g.shape[0], -1)
        grads = []
        d = self.d
        for k in range(self.N):
            Ek = E.reshape(d**k, d, -1)
            left = A[k - 1] if k > 0 else np.ones((1, 1))
            gk = np.einsum("lsr,la,br->sab", Ek, left, Bs[k + 1])
            if k == 0:
                gk = gk[:, 0, :]
            grads.append(gk.reshape(-1))
        return 0.5 * float(E @ E), np.concatenate(grads), E
