# Copyright © 2023 Apple Inc.

import time

import mlx.core as mx

num_features = 1
num_examples = 211_680
num_iters = 1_000
lr = 0.1

# True parameters
w_star = mx.random.normal((num_features,))

# Input examples
X = mx.random.normal((num_examples, num_features))

# Labels
y = (X @ w_star) > 0
print(
    f"X shape {X.shape} "
    f"y shape {y.shape}"
)

# Initialize random parameters
w = 1e-2 * mx.random.normal((num_features,))


def loss_fn(w):
    logits = X @ w
    return mx.mean(mx.logaddexp(0.0, logits) - y * logits)

mx.set_default_device(mx.cpu)
grad_fn = mx.grad(loss_fn)

tic = time.time()
for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    mx.eval(w)

toc = time.time()

loss = loss_fn(w)
final_preds = (X @ w) > 0
acc = mx.mean(final_preds == y)

throughput = num_iters / (toc - tic)
time_taken = toc - tic
print(
    f"Loss {loss.item():.5f}, Accuracy {acc.item():.5f} "
    f"Throughput {throughput:.5f} (it/s) "
    f"Time taken for {num_examples} examples {time_taken:.5f} (s)"
)