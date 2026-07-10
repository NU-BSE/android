import numpy as np

from drlqn.autodiff import T, Param, Linear, MLP, LayerNorm, huber


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        x[i] += eps
        fp = f()
        x[i] -= 2 * eps
        fm = f()
        x[i] += eps
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def check(build, param):
    out = build()
    out.backward()
    ana = param.grad.copy()
    num = numgrad(lambda: build().data.item() if build().data.size == 1
                  else float(build().data.sum()), param.data)
    assert np.allclose(ana, num, atol=1e-4), f"max err {np.abs(ana-num).max()}"


def test_matmul_softmax_grad():
    rng = np.random.default_rng(0)
    W = Param(rng.normal(size=(4, 5)))
    x = T(rng.normal(size=(3, 4)))

    def build():
        return ((x @ W).softmax(axis=-1) * T(rng2)).sum()

    rng2 = np.random.default_rng(1).normal(size=(3, 5))
    check(build, W)


def test_mlp_layernorm_grad():
    rng = np.random.default_rng(2)
    mlp = MLP([6, 8, 3], rng)
    ln = LayerNorm(6)
    x = T(rng.normal(size=(4, 6)))

    def build():
        return (mlp(ln(x)).tanh() ** 2).sum()

    for p in mlp.parameters() + ln.parameters():
        p.grad = None
    check(build, mlp.layers[0].W)
    for p in mlp.parameters() + ln.parameters():
        p.grad = None
    check(build, ln.g)


def test_getitem_concat_grad():
    rng = np.random.default_rng(3)
    W = Param(rng.normal(size=(10, 4)))
    idx = np.array([1, 3, 3, 7])

    def build():
        picked = W[idx]
        return (T.concat([picked, picked], axis=-1) ** 2).sum()

    check(build, W)


def test_huber_grad():
    rng = np.random.default_rng(4)
    W = Param(rng.normal(size=(5,)) * 3)

    def build():
        return huber(W * 1.0)

    check(build, W)


if __name__ == "__main__":
    test_matmul_softmax_grad()
    test_mlp_layernorm_grad()
    test_getitem_concat_grad()
    test_huber_grad()
    print("autodiff gradcheck OK")
