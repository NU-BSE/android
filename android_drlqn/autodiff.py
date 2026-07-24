"""Minimal reverse-mode autodiff engine over numpy arrays.

PyPI (and therefore PyTorch) is unreachable from this sandbox, so DRLQN v0
uses this ~250-line engine instead of torch. It supports everything the
DRLQN-S model needs: matmul, broadcast add/mul, relu/tanh/sigmoid, exp/log,
softmax, reductions, reshape/transpose/concat/slicing and row gather.

Forward-mode (JVP) for the lambda-executor is hand written in
`drlqn/runtime/jvp.py` because the executor graph is small and fixed
(the spec's torch.func.jvp path, replaced by an exact equivalent).
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum grad so that it matches `shape` (inverse of numpy broadcasting)."""
    if grad.shape == shape:
        return grad
    # sum leading dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class T:
    """Tensor node."""

    __slots__ = ("data", "grad", "parents", "_backward", "requires_grad")
    __array_priority__ = 100.0

    def __init__(self, data, parents=(), backward=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.parents = parents
        self._backward = backward
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    # ------------------------------------------------------------------ util
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def detach(self):
        return T(self.data.copy())

    def numpy(self):
        return self.data

    # ------------------------------------------------------------- arithmetic
    @staticmethod
    def _wrap(x):
        return x if isinstance(x, T) else T(x)

    def __add__(self, other):
        other = T._wrap(other)
        out = T(self.data + other.data, (self, other))

        def bw(g):
            return (_unbroadcast(g, self.shape), _unbroadcast(g, other.shape))

        out._backward = bw
        return out

    __radd__ = __add__

    def __neg__(self):
        out = T(-self.data, (self,))
        out._backward = lambda g: (-g,)
        return out

    def __sub__(self, other):
        return self + (-T._wrap(other))

    def __rsub__(self, other):
        return T._wrap(other) + (-self)

    def __mul__(self, other):
        other = T._wrap(other)
        out = T(self.data * other.data, (self, other))

        def bw(g):
            return (
                _unbroadcast(g * other.data, self.shape),
                _unbroadcast(g * self.data, other.shape),
            )

        out._backward = bw
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = T._wrap(other)
        out = T(self.data / other.data, (self, other))

        def bw(g):
            return (
                _unbroadcast(g / other.data, self.shape),
                _unbroadcast(-g * self.data / (other.data ** 2), other.shape),
            )

        out._backward = bw
        return out

    def __matmul__(self, other):
        other = T._wrap(other)
        out = T(self.data @ other.data, (self, other))
        a, b = self.data, other.data

        def bw(g):
            if a.ndim == 1 and b.ndim == 2:      # vector @ matrix -> vector
                return (g @ b.T, np.outer(a, g))
            if a.ndim == 2 and b.ndim == 1:      # matrix @ vector -> vector
                return (np.outer(g, b), a.T @ g)
            ga = g @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ g
            return (_unbroadcast(ga, self.shape), _unbroadcast(gb, other.shape))

        out._backward = bw
        return out

    def __pow__(self, p: float):
        out = T(self.data ** p, (self,))
        out._backward = lambda g: (g * p * self.data ** (p - 1),)
        return out

    # ------------------------------------------------------------ nonlinearity
    def relu(self):
        mask = self.data > 0
        out = T(self.data * mask, (self,))
        out._backward = lambda g: (g * mask,)
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = T(t, (self,))
        out._backward = lambda g: (g * (1 - t * t),)
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = T(s, (self,))
        out._backward = lambda g: (g * s * (1 - s),)
        return out

    def exp(self):
        e = np.exp(self.data)
        out = T(e, (self,))
        out._backward = lambda g: (g * e,)
        return out

    def log(self):
        out = T(np.log(self.data), (self,))
        out._backward = lambda g: (g / self.data,)
        return out

    def softmax(self, axis=-1):
        x = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(x)
        s = e / e.sum(axis=axis, keepdims=True)
        out = T(s, (self,))

        def bw(g):
            dot = (g * s).sum(axis=axis, keepdims=True)
            return (s * (g - dot),)

        out._backward = bw
        return out

    # -------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def bw(g):
            g = np.asarray(g)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, self.shape).copy(),)

        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    # ------------------------------------------------------------------ shape
    def reshape(self, *shape):
        out = T(self.data.reshape(*shape), (self,))
        out._backward = lambda g: (g.reshape(self.shape),)
        return out

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.ndim)))
        inv = np.argsort(axes)
        out = T(self.data.transpose(axes), (self,))
        out._backward = lambda g: (g.transpose(inv),)
        return out

    def swap(self, a, b):
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    def __getitem__(self, idx):
        out = T(self.data[idx], (self,))

        def bw(g):
            full = np.zeros_like(self.data)
            np.add.at(full, idx, g)
            return (full,)

        out._backward = bw
        return out

    def gather_rows(self, index: np.ndarray):
        """index into axis 0 (with repeats allowed)."""
        return self[index]

    @staticmethod
    def concat(tensors, axis=-1):
        tensors = [T._wrap(t) for t in tensors]
        out = T(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors))
        sizes = [t.data.shape[axis] for t in tensors]
        splits = np.cumsum(sizes)[:-1]

        def bw(g):
            return tuple(np.split(g, splits, axis=axis))

        out._backward = bw
        return out

    @staticmethod
    def stack(tensors, axis=0):
        tensors = [T._wrap(t) for t in tensors]
        out = T(np.stack([t.data for t in tensors], axis=axis), tuple(tensors))

        def bw(g):
            return tuple(np.moveaxis(g, axis, 0))

        out._backward = bw
        return out

    # --------------------------------------------------------------- backward
    def backward(self, grad=None):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen or not v.requires_grad:
                return
            seen.add(id(v))
            for p in v.parents:
                build(p)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        for v in reversed(topo):
            if v._backward is None or v.grad is None:
                continue
            grads = v._backward(v.grad)
            for p, g in zip(v.parents, grads):
                if not p.requires_grad or g is None:
                    continue
                if p.grad is None:
                    p.grad = np.zeros_like(p.data)
                p.grad += g


# ---------------------------------------------------------------------- layers
class Param(T):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def parameters(self):
        out = []
        for v in self.__dict__.values():
            if isinstance(v, Param):
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Module):
                        out.extend(x.parameters())
                    elif isinstance(x, Param):
                        out.append(x)
        return out

    def n_params(self):
        return sum(p.data.size for p in self.parameters())

    def state_dict(self):
        return [p.data.copy() for p in self.parameters()]

    def load_state_dict(self, sd):
        for p, d in zip(self.parameters(), sd):
            p.data = d.copy()


class Linear(Module):
    def __init__(self, d_in, d_out, rng, bias=True, scale=None):
        s = scale if scale is not None else np.sqrt(2.0 / d_in)
        self.W = Param(rng.normal(0, s, size=(d_in, d_out)))
        self.b = Param(np.zeros(d_out)) if bias else None

    def __call__(self, x: T) -> T:
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y


class MLP(Module):
    def __init__(self, dims, rng, act="relu"):
        self.layers = [Linear(dims[i], dims[i + 1], rng) for i in range(len(dims) - 1)]
        self.act = act

    def __call__(self, x: T) -> T:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu() if self.act == "relu" else x.tanh()
        return x


class LayerNorm(Module):
    def __init__(self, d):
        self.g = Param(np.ones(d))
        self.beta = Param(np.zeros(d))

    def __call__(self, x: T) -> T:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        inv = (var + 1e-5) ** -0.5
        return xc * inv * self.g + self.beta


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, clip=5.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, betas[0], betas[1], eps, clip
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        # global grad-norm clip
        sq = 0.0
        for p in self.params:
            if p.grad is not None:
                sq += float((p.grad ** 2).sum())
        norm = np.sqrt(sq)
        scale = min(1.0, self.clip / (norm + 1e-12))
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


def huber(x: T, delta: float = 1.0) -> T:
    """Elementwise Huber of residual x, returns mean scalar."""
    a = np.abs(x.data)
    quad = a <= delta
    # value = 0.5 x^2 (quad) ; delta*(|x|-0.5 delta) (lin)
    out = T(np.where(quad, 0.5 * x.data ** 2, delta * (a - 0.5 * delta)), (x,))

    def bw(g):
        return (g * np.where(quad, x.data, delta * np.sign(x.data)),)

    out._backward = bw
    return out.mean()
