"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.array(x, dtype=float)
    for idx in np.ndindex(x.shape):
        x_org = x[idx]
        x[idx] = x_org + eps
        f_plus = f(x)
        x[idx] = x_org - eps
        f_minus = f(x)
        x[idx] = x_org
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # Return max relative error between analytic and numeric gradients.
    a_arr = np.asarray(analytic_grad, dtype=float)
    n_arr = np.asarray(numeric_grad, dtype=float)
    diff = np.abs(a_arr - n_arr)
    max_arr = np.maximum(
        np.maximum(a_arr, n_arr),
        tol
    )
    e_arr = diff / max_arr
    return float(np.max(e_arr))

# Step 3 - make_dense
import copy

def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # your approach here
    (W, b) = weight_init_fn(in_dim, out_dim)

    def forward(x):
      y = x @ W + b
      cache = (copy.deepcopy(x), copy.deepcopy(W))
      return y, cache

    def backward(dout, cache):
      (x, W) = cache
      dW = x.T @ dout
      db = np.sum(dout, axis=0)
      dx = dout @ W.T
      return dx, {"W": dW, "b": db}

    return {
      "params": {
        "W": W,
        "b": b
      },
      "forward": forward,
      "backward": backward
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # your approach here
    match kind:
      case "relu":
        params = {}

        def forward(x):
          mask = (x > 0).astype(x.dtype)
          y = x * mask
          cache = mask
          return y, cache

        def backward(dout, cache):
          dx = dout * cache
          return dx, {}

      case "tanh":
        params = {}

        def forward(x):
          y = np.tanh(x)
          cache = copy.deepcopy(y)
          return y, cache

        def backward(dout, cache):
          dx = dout * (1 - cache ** 2)
          return dx, {}

      case "sigmoid":
        params = {}

        def forward(x):
          y = 1 / (1 + np.exp(-x))
          cache = copy.deepcopy(y)
          return y, cache

        def backward(dout, cache):
          dx = dout * cache * (1 - cache)
          return dx, {}
      
      case _:
        raise ValueError(f"Unknown activation: {kind}")

    return {
      "params": params,
      "forward": forward,
      "backward": backward
    }

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

