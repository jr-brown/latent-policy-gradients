# Preference Modelling Models

This module provides phenomenological models for how agent preferences evolve during training. These models are fit to empirical preference data (pairwise goal comparisons) to understand the computational mechanisms underlying preference formation.

## Class Hierarchy

```
PhenomenologicalModel (ABC)
├── SimpleWeightsModel (ABC)
│   ├── ClosedFormModel (ABC)
│   │   ├── RWModel (Rescorla-Wagner)
│   │   └── KLModel (KL-Divergence)
│   ├── MultiChoiceKLModel
│   └── DiagonalQuadraticMultiChoiceKLModel
└── ShardedMultiChoiceKLModel (Contextual Shards)

Mixins:
├── PipelineModeMixin (sequential/simultaneous/memoryless pipeline modes)
├── SaliencyModelMixin (theta matrix for gradient saliency)
└── NonLinearityMixin (optional non-linearity on weights)

Value Functions ((w, φ) → scalar):
├── LinearValueFunction (default): value = (S @ w) · φ
├── MLPValueFunction: value = MLP(concat(w, φ))
└── BilinearValueFunction: value = σ(Bφ)ᵀ A σ(Cw)
```

## Model Types

Models are accessed via the registry in `__init__.py`:

| `model_type` | Class | Description |
|--------------|-------|-------------|
| `"rw"` | `RWModel` | Rescorla-Wagner with saliency |
| `"kl"` | `KLModel` | KL-divergence with closed-form solution |
| `"multi_kl"` | `MultiChoiceKLModel` | Multi-choice KL with native saliency matrix S |
| `"sharded_multi_kl"` | `ShardedMultiChoiceKLModel` | Contextual sharded model with K sub-policies |
| `"diagonal_quadratic_multi_kl"` | `DiagonalQuadraticMultiChoiceKLModel` | Quadratic feature expansion with diagonal saliency |

## Core Concepts

### Weight Evolution

All models track a weight vector `w` (or structured parameters for sharded models) that evolves through training stages. The weight update follows:

```
dw/dt = gradient_fn(w, environment)
```

Integration is performed numerically (configurable via `integration_kwargs`) or analytically for closed-form models.

### Saliency

Two saliency paradigms exist:

1. **Gradient Saliency** (`theta` matrix, used in `SaliencyModelMixin`):
   - Applied to gradients: `dw/dt = -theta @ grad_loss(w)`
   - Modulates which features learn from which training signals

2. **Native Saliency** (via `ValueFunction`, used in `MultiChoiceKLModel` and `ShardedMultiChoiceKLModel`):
   - Applied to weights in choice computation via value functions
   - Directly transforms learned weights into goal values

### Value Functions

Value functions compute goal values from agent weights and features: `(w, φ) → scalar`. All models use a unified `ValueFunction` interface:

1. **LinearValueFunction** (default): `value = (S @ w) · φ`
   - Supports structure constraints: `full`, `upper_triangular`, `diagonal`
   - S matrix is learnable, initialized to identity by default
2. **MLPValueFunction**: `value = MLP(concat(w, φ))`
3. **BilinearValueFunction**: `value = σ(Bφ)ᵀ A σ(Cw)`

Configure via `value_function` parameter in model construction. If not specified, defaults to `LinearValueFunction()` with a learnable S matrix initialized to identity.

### Pipeline Modes

Models support different ways of processing multi-stage training pipelines:

| Mode | Description |
|------|-------------|
| `"sequential"` | Process stages in order (default) |
| `"simultaneous"` | Flatten all stages into one (no temporal ordering) |
| `"memoryless"` | Use only the final stage (ignores history) |

### S Matrix Structure Constraints

The native saliency matrix S can be constrained:

| Structure | Description |
|-----------|-------------|
| `"full"` | No restriction (default) |
| `"upper_triangular"` | Only upper triangular entries |
| `"diagonal"` | Only diagonal entries (per-feature scaling) |

### S Matrix Initialization

The S matrix can be initialized differently:

| Init | Description |
|------|-------------|
| `"identity"` | Identity matrix (default) |
| `"random_gaussian"` | i.i.d. N(0,1) values respecting structure constraint |

For random initialization, use `s_matrix_init_seed` to control reproducibility.

---

## Model Details

### RWModel (Rescorla-Wagner)

Classical associative learning with saliency-weighted updates.

**Update rule:**
```
s = theta @ phi
w_new = w + (1 - w·phi) / ||s||_1 * s
```

Where `phi` is the combined goal feature vector (binary indicator of active features).

**Hyperparameters:**
- `beta`: Inverse temperature for choice (not used in equilibrium)
- `theta`: Saliency matrix (n_features × n_features)

---

### KLModel (KL-Divergence)

Gradient descent on KL divergence with closed-form equilibrium.

**Loss:**
```
L = KL(target || policy) = q*log(q/σ) + (1-q)*log((1-q)/(1-σ))
```
where `σ = sigmoid(w·phi)` and `q` is the target goal probability.

**Equilibrium:**
```
w* = α* · (theta @ phi) + w_perp
α* = (logit(q) - w_perp·phi) / (phi^T @ theta @ phi)
```

**Hyperparameters:**
- `q`: Target goal probability (default ~0.95 via logit init 2.944)
- `theta`: Saliency matrix

---

### MultiChoiceKLModel

Multi-choice softmax policy with configurable value functions.

**Policy:**
```
π(φ^i) = exp(value_function(f(w), φ^i)) / (Σ_j exp(value_function(f(w), φ^j)) + 1)
```

Where:
- `w` is the learned weight vector
- `f(w)` is an optional non-linearity
- `value_function` computes goal value from weights and features
- The `+1` in the denominator represents the "null" option

**Value Function Types:**
- `LinearValueFunction` (default): `value = (S @ w) · φ`
- `MLPValueFunction`: `value = MLP(concat(w, φ))`
- `BilinearValueFunction`: `value = σ(Bφ)ᵀ A σ(Cw)`

**Loss:** KL divergence between total goal probability and target q.

**KL Direction:**
- `reverse_kl=True` (default): KL(policy || target) - mode-seeking
- `reverse_kl=False`: KL(target || policy) - mean-seeking

**Hyperparameters:**
- `q`: Target goal probability
- Value function parameters (S matrix for linear, MLP weights, or bilinear B/C/A)
- Non-linearity params if applicable (e.g., `gamma` for power_law)

**Constructor options:**
- `value_function`: ValueFunction instance or config dict. Defaults to LinearValueFunction with full structure.
  - `None` or `{}`: Linear with learnable S matrix initialized to identity
  - `{"structure": "diagonal"}`: Diagonal saliency
  - `{"type": "mlp", "hidden_sizes": [32]}`: MLP value function
  - `{"type": "bilinear", "rank": 16}`: Bilinear value function
- `pipeline_mode`: `"sequential"`, `"simultaneous"`, `"memoryless"`
- `non_linearity`: `"none"`, `"quadratic"`, `"sigmoid"`, `"power_law"`

---

### DiagonalQuadraticMultiChoiceKLModel

Efficient quadratic feature expansion with diagonal saliency.

Expands features from n to n + n² (original features plus all pairwise products), but uses only a diagonal saliency vector instead of a full matrix:

```
φ_expanded = [φ, flatten(φ ⊗ φ)]
effective_w = S_diag * f(w)  # element-wise, not matmul
```

This is O(n + n²) per step instead of O((n + n²)²), making it ~100x faster for n=10 features.

**Hyperparameters:**
- `q`: Target goal probability
- `S_diag`: Diagonal saliency vector (n + n² elements)
- `pair_saliency_init`: Initial saliency for feature-pair terms (default 0.01)

---

## Value Functions

Value functions map `(w, φ) → scalar` and are configured via the `value_function` parameter in model construction.

### LinearValueFunction

Linear value function that applies a saliency matrix to weights.

**Forward pass:**
```
effective_w = S @ w
value = effective_w · φ
```

**Structure constraints:**
- `"full"`: Unconstrained S matrix (default)
- `"upper_triangular"`: Only upper triangular entries
- `"diagonal"`: Only diagonal entries (per-feature scaling)

**Constructor options:**
- `structure`: `"full"`, `"upper_triangular"`, `"diagonal"`
- `init`: `"identity"`, `"random_gaussian"`
- `seed`: Random seed for Gaussian init
- `latent_dimension`: Dimension of weight vector (allows non-square S)

**Config example:**
```yaml
value_function:
  type: "linear"
  structure: "diagonal"
```

---

### MLPValueFunction

MLP-based value function that learns arbitrary non-linear interactions between weights and features.

**Architecture:**
```
input: concat(w, φ)  →  hidden layers  →  scalar value
       (2n dims)         (configurable)      (1 dim)
```

**Forward pass:**
```
x = concat(w, φ)
for layer in hidden_layers:
    x = activation(W @ x + b)
value = W_final @ x + b_final
```

**Constructor options:**
- `hidden_sizes`: List of hidden layer sizes (default: `[32]`)
- `activation`: `"relu"`, `"tanh"`, `"gelu"`, `"silu"` (default: `"silu"`)
- `seed`: Random seed for weight initialization
- `l1_weight`: L1 regularization strength for sparsity (default: 0.0)

**Config example:**
```yaml
value_function:
  type: "mlp"
  hidden_sizes: [64, 32]
  activation: "silu"
  l1_weight: 0.001
```

---

### BilinearValueFunction

Factored bilinear form with optional activation and configurable A structure.

**General form:**
```
value = σ(B @ φ)ᵀ A σ(C @ w)
```

Where A can be:
- `A_structure="none"`: No A, simple dot product: `value = σ(Bφ) · σ(Cw)`
- `A_structure="diagonal"`: A is a vector (k,): `value = A · (σ(Bφ) ⊙ σ(Cw))`
- `A_structure="full"`: A is a matrix (k,k): `value = σ(Bφ)ᵀ A σ(Cw)`

The effective bilinear matrix (without activation) is:
- `A_structure="none"`: M = Bᵀ C
- `A_structure="diagonal"`: M = Bᵀ diag(A) C
- `A_structure="full"`: M = Bᵀ A C

The activation makes gradients depend on `w`, allowing the gradient subspace to change as `w` evolves. This enables exploration beyond the fixed n_features gradient subspace limit.

**Parameters:**
- `B`: (k, n_features) - projects features
- `C`: (k, n_features) - projects weights
- `A`: depends on A_structure:
  - `"none"`: no A parameter
  - `"diagonal"`: (k,) vector
  - `"full"`: (k, k) matrix

**Initialization:** Starts as approximate identity (`value ≈ w · φ`) with B≈I, C≈I, A≈I or A≈1.

**Constructor options:**
- `rank`: Dimension of intermediate space k (default: `n_features`)
- `activation`: `"none"`, `"relu"`, `"silu"` (default: `"silu"`)
- `A_structure`: Structure of A parameter (default: `"diagonal"`):
  - `"none"`: No A, simple dot product of projections
  - `"diagonal"`: A is a vector (element-wise weighting)
  - `"full"`: A is a matrix (full bilinear form over projections)
- `seed`: Random seed for initialization
- `l1_weight`: L1 regularization strength (default: 0.0)
- `init_scale`: Scale for random initialization of extra dimensions when rank > n_features (default: 0.01)

**Config example (diagonal A, default):**
```yaml
value_function:
  type: "bilinear"
  rank: 16
```

**Config example (full matrix A):**
```yaml
value_function:
  type: "bilinear"
  rank: 16
  A_structure: "full"
```

**Config example (no A):**
```yaml
value_function:
  type: "bilinear"
  rank: 16
  A_structure: "none"
```

**Comparison with other approaches:**

| Approach | Structure | Parameters | Gradient dynamics |
|----------|-----------|------------|-------------------|
| Matrix saliency | `φ · (S @ w)` | O(n²) | Fixed subspace |
| MLP | Learned | O(n × hidden) | Arbitrary |
| Bilinear (A=none) | `(Bφ) · (Cw)` | O(2kn) | Fixed subspace |
| Bilinear (A=diag) | `a · (Bφ ⊙ Cw)` | O(2kn + k) | Fixed subspace |
| Bilinear (A=full) | `(Bφ)ᵀ A (Cw)` | O(2kn + k²) | Fixed subspace |
| + activation | `σ(Bφ)ᵀ A σ(Cw)` | same | w-dependent |

---

### ShardedMultiChoiceKLModel (Contextual Shards)

Multiple sub-policies ("shards") with context-dependent mixing.

The agent has K shards, each with:
- **Value weights** `v^(k)`: Determine preferences over objects (zero-initialized)
- **Activation weights** `a^(k)`: Determine context-dependent shard activation
- **Activation bias** `α_k`: Context-independent activation component

**Shard probabilities:**
```
φ_ctx = Σ_i φ^i  # sum of all object features (context)
p_k = softmax(s_a * (a^(k) · φ_ctx) + s_b * α_k)
```

**Aggregate policy:**
```
π_θ(φ^i) = Σ_k p_k * π^(k)(φ^i)
```

Where each shard policy computes values using the value function:
```
π^(k)(φ^i) = exp(value_function(f(s_v * v^(k)), φ^i)) / Z
```

**Value Function Types:**
- `LinearValueFunction` (default): `value = (S @ w) · φ`
- `MLPValueFunction`: `value = MLP(concat(w, φ))`
- `BilinearValueFunction`: `value = σ(Bφ)ᵀ A σ(Cw)`

**Hyperparameters:**
- `q`: Target goal probability
- `gamma`: Power transform for activation weight init sparsity
- `activation_weight_init_scale`: Scale for activation weight initialization
- `value_scale` (`s_v`): Scale factor for value weights
- `activation_weight_scale` (`s_a`): Scale for context activations
- `activation_bias_scale` (`s_b`): Scale for bias activations
- Value function params (S matrix for linear, MLP weights, or bilinear B/C/A)

**Constructor options:**
- `n_shards`: Number of shards K (default 2)
- `value_function`: ValueFunction instance or config dict. Defaults to LinearValueFunction with full structure.
  - `None` or `{}`: Linear with learnable S matrix initialized to identity
  - `{"structure": "diagonal"}`: Diagonal saliency
  - `{"type": "mlp", "hidden_sizes": [32]}`: MLP value function
  - `{"type": "bilinear", "rank": 16}`: Bilinear value function
- `pipeline_mode`: `"sequential"`, `"simultaneous"`, `"memoryless"`
- `non_linearity`: `"none"`, `"quadratic"`, `"sigmoid"`, `"power_law"`

**Config example with bilinear value function:**
```yaml
function_kwargs:
  model_type: "contextual_sharded_multi_kl"
  model_kwargs:
    n_shards: 2
    value_function:
      type: "bilinear"
      rank: 16
```

**Config example with MLP value function:**
```yaml
function_kwargs:
  model_type: "contextual_sharded_multi_kl"
  model_kwargs:
    n_shards: 2
    value_function:
      type: "mlp"
      hidden_sizes: [16]
      l1_weight: 0.01
```

---

## Non-Linearity Options

Available via `NonLinearityMixin`:

| Option | Function | Notes |
|--------|----------|-------|
| `"none"` | `f(w) = w` | Default, identity |
| `"quadratic"` | `f(w) = w²` | Requires non-zero init |
| `"sigmoid"` | `f(w) = sigmoid(w)` | Bounded output |
| `"power_law"` | `f(w) = sign(w) * |w|^γ` | Learnable γ, requires non-zero init |

For `quadratic` and `power_law`, weights are initialized with small non-zero values (controlled by `init_std` hyperparameter) since `f'(0) = 0`.

---

## Usage Example

```python
from src.preference_modelling.models import get_model

# Default model (linear value function with learnable S matrix)
model = get_model("multi_kl",
    integration_kwargs={"max_steps": 100}
)

# Model with diagonal saliency
model = get_model("multi_kl",
    value_function={"structure": "diagonal"},
    integration_kwargs={"max_steps": 100}
)

# Model with MLP value function
model = get_model("multi_kl",
    value_function={
        "type": "mlp",
        "hidden_sizes": [64, 32],
        "activation": "silu",
    },
    integration_kwargs={"max_steps": 100}
)

# Model with bilinear value function
model = get_model("multi_kl",
    value_function={
        "type": "bilinear",
        "rank": 16,
        "activation": "silu",
    },
    integration_kwargs={"max_steps": 100}
)

# Sharded model (default linear value function with learnable S matrix)
model = get_model("sharded_multi_kl",
    n_shards=4,
    pipeline_mode="sequential",
    integration_kwargs={"max_steps": 64}
)

# Sharded model with bilinear value function
model = get_model("sharded_multi_kl",
    n_shards=2,
    value_function={
        "type": "bilinear",
        "rank": 16,
    },
    integration_kwargs={"max_steps": 100}
)

# Initialize parameters
params = model.init_params_from_spec(n_features=10)

# Compute loss for a training example
loss = model.single_example_loss(params, padded_pipeline, goal_0, goal_1, prob_0, prob_1, weight)
```

---

## Config Examples

### Linear Value Function (default)

```yaml
function_kwargs:
  model_type: "multi_kl"
  model_kwargs:
    reverse_kl: true
    pipeline_mode: "sequential"
    integration_kwargs:
      max_steps: 100
      step_size: 0.1
```

### Linear with Diagonal Saliency

```yaml
function_kwargs:
  model_type: "multi_kl"
  model_kwargs:
    value_function:
      structure: "diagonal"
      init: "random_gaussian"
      seed: 42
    reverse_kl: true
    pipeline_mode: "sequential"
    integration_kwargs:
      max_steps: 100
```

### MLP Value Function

```yaml
function_kwargs:
  model_type: "multi_kl"
  model_kwargs:
    value_function:
      type: "mlp"
      hidden_sizes: [64, 32]
      activation: "silu"
      l1_weight: 0.001
    pipeline_mode: "sequential"
    integration_kwargs:
      max_steps: 100
```

### Bilinear Value Function

```yaml
function_kwargs:
  model_type: "multi_kl"
  model_kwargs:
    value_function:
      type: "bilinear"
      rank: 16
    pipeline_mode: "sequential"
    integration_kwargs:
      max_steps: 100
```

### Sharded Model with Bilinear Value Function

```yaml
function_kwargs:
  model_type: "contextual_sharded_multi_kl"
  model_kwargs:
    n_shards: 2
    value_function:
      type: "bilinear"
      rank: 16
```

### Sharded Model with MLP Value Function

```yaml
function_kwargs:
  model_type: "contextual_sharded_multi_kl"
  model_kwargs:
    n_shards: 2
    value_function:
      type: "mlp"
      hidden_sizes: [16]
      l1_weight: 0.01
```

---

## File Structure

| File | Contents |
|------|----------|
| `base.py` | Abstract base classes, mixins, utility functions |
| `closed_form.py` | `ClosedFormModel`, `RWModel`, `KLModel` |
| `multi_kl.py` | `MultiChoiceKLModel`, `DiagonalQuadraticMultiChoiceKLModel` |
| `saliency.py` | `ValueFunction`, `LinearValueFunction`, `MLPValueFunction`, `BilinearValueFunction`, `make_value_function()` |
| `sharded.py` | `ShardedMultiChoiceKLModel` |
| `__init__.py` | Model registry and exports |
