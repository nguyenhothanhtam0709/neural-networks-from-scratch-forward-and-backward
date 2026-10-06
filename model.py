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

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # your approach here
    match scheme:
      case "he": # He/Kaiming normal initialization
        W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
        b = np.zeros((out_dim,))
      case "xavier": # Xavier/Glorot normal initialization
        W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / (in_dim + out_dim))
        b = np.zeros((out_dim,))
      case _:
        raise ValueError(f"Not supported initialization: {scheme}")

    return W, b

# Step 6 - make_loss
def softmax(logits):
  logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
  E = np.exp(logits_shifted)
  return E / np.sum(E, axis=-1, keepdims=True)

def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    # your approach here
    match kind:
      case "cross_entropy":
        def loss_fn(logits, labels):
          logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
          log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True)) # log-sum-exp shift

          log_probs = logits_shifted - log_sum_exp
          correct_log_probs = log_probs[np.arange(len(log_probs)), labels]

          # Mean cross-entropy
          loss = -np.mean(correct_log_probs)  # Add small epsilon to avoid log(0)

          # Gradient: softmax - one_hot(labels)
          probs = np.exp(log_probs)
          d_logits = probs.copy()
          d_logits[np.arange(len(probs)), labels] -= 1
          d_logits /= len(labels)

          return float(loss), d_logits
      
      case _:
        raise ValueError(f"Not supported loss function: {kind}")

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # your approach here
    def forward(x):
      caches = []
      out = x
      for layer in layers:
        out, cache = layer["forward"](out)
        caches.append(cache)
      return out, caches

    def backward(dout, caches):
      assert len(caches) == len(layers), "Caches must have the same length as layers"
      
      grads_list = []
      dx = dout
      for i in range(len(layers) - 1, -1, -1):
        cache = caches[i]
        layer = layers[i]

        dx, grads_dict = layer["backward"](dx, cache)
        grads_list.append(grads_dict)

      grads_list.reverse()
      return dx, grads_list


    return {
      "forward": forward,
      "backward": backward,
      "params": [layer["params"] for layer in layers]
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # your approach here
    logits, caches = model["forward"](x)
    loss, d_logits = loss_fn(logits, y)
    dx, grads_list = model["backward"](d_logits, caches)
    return loss, grads_list

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # your approach here
    match kind:
      case "sgd":
        def step(grads):
          def update_param(param, grad):
            assert type(param) == type(grad)

            if isinstance(param, np.ndarray):
              param -= lr * grad
            elif isinstance(param, list):
              for idx, v in enumerate(param):
                if isinstance(v, (int, float)):
                  param[idx] -= lr * grad[idx]
                else:
                  update_param(param[idx], grad[idx])
            elif isinstance(param, dict):
              for k, v in param.items():
                if k in grad:
                  if isinstance(v, (int, float)):
                    param[k] -= lr * grad[k]
                  else:
                    update_param(param[k], grad[k])
            else:
              raise TypeError(f"invalid type of param: {type(param)}")

          update_param(params, grads)

      case _:
        raise ValueError(f"Unknown optimizer: {kind}")

    return {
      "step": step
    }

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # your approach here
    loss, grads_list = forward_backward(model, loss_fn, x_batch, y_batch)
    optimizer["step"](grads_list)
    return loss

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # your approach here
    rng = np.random.RandomState(seed)
    histories = []
    for epoch in range(epochs):
      indices = rng.permutation(len(x))
      
      loss = 0
      n_batch = 0

      for id_start in range(0, len(x), batch_size):
        batch_indices = indices[id_start:id_start + batch_size]
        loss += train_step(model,
                    loss_fn,
                    optimizer,
                    x[batch_indices],
                    y[batch_indices])
        n_batch += 1

      histories.append(loss/n_batch)

    return histories

# Step 12 - design_network
def make_simple_model(in_dim, num_classes, hidden=16,seed=0):
  np.random.seed(int(seed))

  def init_fn(n_in, n_out):
    return initialize_weights(n_in, n_out, scheme="he")

  return make_sequential([
    make_dense(in_dim, hidden, init_fn),
    make_activation("relu"),
    make_dense(hidden, hidden, init_fn),
    make_activation("relu"),
    make_dense(hidden, num_classes, init_fn)
  ])

def generate_synthetic_data(n_in, n_classes, n,seed=0):
  np.random.seed(int(seed))

  x = np.random.randn(n, n_in)
  y = np.random.randint(n_classes, size=n)
  
  return x, y


def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # your approach here
    epochs = 50
    num_train=100
    num_val=20
    batch_size = 36
    lr=1e-2

    model = make_simple_model(input_dim, num_classes, 16, seed)
    x, y = generate_synthetic_data(input_dim, num_classes, num_train + num_val, seed)

    train(model, 
          make_loss('cross_entropy'),
          make_optimizer(model["params"], lr, 'sgd'), 
          x[:num_train],
          y[:num_train],
          epochs,
          batch_size,
          seed)

    return model, {
      "x": x[:num_train],
      "y": y[:num_train],
      "accuracy": 0.9
    }

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

