"""Search nested pattern teachers whose predicted saddle sequences differ across MPS/binary/HT."""
import itertools
import sys

sys.path.insert(0, "experiments/locality")
from rules import describe_sequence, get_tree, predicted_sequence  # noqa: E402

N = int(sys.argv[1])
kmax = int(sys.argv[2])
trees = [get_tree(k, N) for k in ("mps", "binary", "ht")]
w = [1.0, 0.6, 0.35]
found = []
sites = range(N)
for S2 in itertools.chain.from_iterable(itertools.combinations(sites, m) for m in range(2, N + 1)):
    for S3 in itertools.chain.from_iterable(itertools.combinations(sites, m) for m in range(2, N + 1)):
        if S2 == S3:
            continue
        for v3 in (1, 2):  # value of U3 on S3 (2 = a third basis vector, needs d = 3)
            p = [[0] * N, [1 if i in S2 else 0 for i in sites], [v3 if i in S3 else 0 for i in sites]]
            seqs = [predicted_sequence(t, p, w) for t in trees]
            ks = [k for s in seqs for (_, k, _, _) in s]
            if max(ks) > kmax or any(c for s in seqs for (*_, c) in s):
                continue
            descs = [describe_sequence(s) for s in seqs]
            nested = set(S2) < set(S3) or set(S3) < set(S2)
            score = len(set(descs)) + (1 if nested else 0)
            found.append((score, nested, S2, S3, v3, descs))
found.sort(key=lambda x: -x[0])
for f in found[:25]:
    print(f[0], "nested" if f[1] else "", "S2", f[2], "S3", f[3], "v3", f[4])
    for name, d in zip(("mps", "bin", "ht"), f[5]):
        print("    ", name, d)
