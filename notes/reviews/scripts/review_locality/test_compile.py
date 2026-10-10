import time, torch
torch.set_num_threads(1)
def f(x):
    return (x**2).sum().sin()
t0 = time.time()
try:
    g = torch.compile(torch.func.grad(f), fullgraph=True, dynamic=False)
    print(g(torch.ones(5, dtype=torch.float64)))
    print("compile ok", time.time() - t0)
except Exception as e:
    print("compile FAILED", type(e).__name__, str(e)[:500], time.time() - t0)
