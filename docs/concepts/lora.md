# LoRA

LoRA (low-rank adaptation) trains a small change to the model. The base weights do not change.

Open this file in a Markdown preview to see the math (VS Code: `Cmd+Shift+V`).

## 1. A linear layer

Most weights of a transformer are in linear layers:

$$
y = W x, \qquad W \in \mathbb{R}^{d_{out} \times d_{in}}
$$

Example: `q_proj` in Ministral 3 8B has $d_{in} = d_{out} = 4096$. That is $4096 \times 4096 = 16{,}777{,}216$ weights.

## 2. Fine-tuning learns a change

Fine-tuning changes the weights:

$$
W' = W + \Delta W
$$

Full fine-tuning trains every entry of $\Delta W$. Then $\Delta W$ has the same size as $W$.

## 3. What training stores for each weight

One training step does three things:

1. **Forward pass**: calculate the loss $L$.
2. **Backward pass**: calculate the gradient $g = \partial L / \partial \theta$ for each trainable weight $\theta$.
3. **Optimizer step**: change each trainable weight.

AdamW keeps two running values for each trainable weight: $m$ (the mean of the gradients) and $v$ (the mean of the squared gradients):

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_{t-1} \right)
$$

Here $\eta$ is the learning rate, $\lambda$ is the weight decay, and $\hat m_t$, $\hat v_t$ are $m_t$, $v_t$ with a bias correction.

In fp32, each number uses 4 bytes:

| Weight type | Weight | Gradient | $m$ | $v$ | Total |
|---|---|---|---|---|---|
| Trainable | 4 B | 4 B | 4 B | 4 B | **16 B** |
| Frozen | 4 B | – | – | – | **4 B** |

A frozen weight has `requires_grad = False`. PyTorch does not calculate a gradient for it. We do not give it to the optimizer, so the optimizer keeps no $m$ and no $v$ for it.

For 8 billion weights:

| | Memory |
|---|---|
| All trainable (full fine-tuning) | $8 \times 10^9 \times 16\,\text{B} = 128$ GB |
| All frozen | $8 \times 10^9 \times 4\,\text{B} = 32$ GB |

One H100 has 80 GB. Full fine-tuning in fp32 does not fit.

**Note.** The backward pass still goes through the frozen layers. The gradient must pass through them to reach the trainable weights in earlier layers. This needs the activations of the forward pass, but not the optimizer values. Gradient checkpointing makes the activation memory smaller (see `training-stages.md`).

## 4. The LoRA idea: a low-rank change

Research found that the change that a task needs usually has a low rank. LoRA writes the change as the product of two thin matrices:

$$
\Delta W = \frac{\alpha}{r}\, B A, \qquad A \in \mathbb{R}^{r \times d_{in}}, \quad B \in \mathbb{R}^{d_{out} \times r}, \quad r \ll d_{in}, d_{out}
$$

The rank of $BA$ is at most $r$. We use $r = 16$.

The forward pass never builds $\Delta W$. It calculates $Ax$ first, which has only $r$ numbers:

$$
y = W x + \frac{\alpha}{r}\, B (A x)
$$

$W$ is frozen. Only $A$ and $B$ are trainable.

**Size.** LoRA trains $r(d_{in} + d_{out})$ numbers instead of $d_{in} d_{out}$:

$$
\text{q\_proj:} \quad 16 \times (4096 + 4096) = 131{,}072 \quad \text{instead of} \quad 16{,}777{,}216 \quad (0.8\,\%)
$$

**LoRA does not approximate the gradient of $W$.** $A$ and $B$ are weights in the forward pass. The output $y$ goes through the next layers to the loss, so the loss depends on $A$ and $B$. The backward pass calculates $\partial L / \partial A$ and $\partial L / \partial B$ with the chain rule, as for all weights. It does not calculate the gradient of $W$. (A different method, GaLore, trains $W$ and compresses its gradient. Jevstral does not use it.)

### A small example

Let $d_{in} = d_{out} = 2$, $r = 1$, $\alpha / r = 2$, and $L = \lVert y - t \rVert^2$:

$$
x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad
W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
A = \begin{bmatrix} 0.5 & -0.3 \end{bmatrix}, \quad
B = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad
t = \begin{bmatrix} 2 \\ 2 \end{bmatrix}
$$

Step 0:

$$
A x = 0.5 \cdot 1 - 0.3 \cdot 2 = -0.1, \qquad
y = W x + 2\, B (A x) = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \qquad
L = (1 - 2)^2 + (2 - 2)^2 = 1
$$

If we change $B$ to $\begin{bmatrix} 1 & 0 \end{bmatrix}^\top$, then $y = \begin{bmatrix} 0.8 & 2 \end{bmatrix}^\top$ and $L = 1.44$. The loss changes when $B$ changes, so the loss depends on $B$.

The gradients at step 0, with $\delta = \partial L / \partial y = 2 (y - t) = \begin{bmatrix} -2 & 0 \end{bmatrix}^\top$ (formulas in section 5):

$$
\frac{\partial L}{\partial B} = 2\, \delta\, (A x) = 2 \begin{bmatrix} -2 \\ 0 \end{bmatrix} (-0.1) = \begin{bmatrix} 0.4 \\ 0 \end{bmatrix}, \qquad
\frac{\partial L}{\partial A} = 2\, B^\top \delta\, x^\top = \begin{bmatrix} 0 & 0 \end{bmatrix}
$$

### Compute

**Forward pass.** LoRA adds $B(Ax)$ to $Wx$. For `q_proj`, for each token:

| Calculation | Multiplications |
|---|---|
| $W x$ | $4096 \times 4096 = 16{,}777{,}216$ |
| $A x$ | $16 \times 4096 = 65{,}536$ |
| $B (A x)$ | $4096 \times 16 = 65{,}536$ |

The LoRA path adds 0.8 %. Calculate $B(Ax)$, not $(BA)x$: the product $BA$ is a full $4096 \times 4096$ matrix and costs $16 \times 4096 \times 4096$ multiplications.

**Backward pass.** For a frozen layer, the backward pass calculates the input gradient $W^\top \delta$ (to reach earlier layers), but not the weight gradient $\delta x^\top$. Approximately, a full fine-tuning step costs 3 forward passes and a LoRA step costs 2 forward passes and a small amount more.

**Inference.** After training, calculate $W' = W + \frac{\alpha}{r} B A$ one time. Then $y = W' x$ costs the same as the base model.

## 5. Initialization: $B = 0$, $A$ random

LoRA starts with:

- $A$: small random values.
- $B$: all zeros.

**Result at step 0.** For any $A$:

$$
B = 0 \;\Rightarrow\; BA = 0 \;\Rightarrow\; W' = W + 0 = W
$$

The random values in $A$ have no effect, because a product with a zero matrix is zero. At step 0, the LoRA model gives exactly the same outputs as the base model. Training starts from the knowledge of the base model, not from noise.

**Why is $A$ random and not also zero?** Look at the gradients. Let $h = A x$ and $\delta = \partial L / \partial y$:

$$
\frac{\partial L}{\partial B} = \frac{\alpha}{r}\, \delta\, h^\top = \frac{\alpha}{r}\, \delta\, (A x)^\top
$$

$$
\frac{\partial L}{\partial A} = \frac{\alpha}{r}\, B^\top \delta\, x^\top
$$

| Start | $\partial L / \partial B$ | $\partial L / \partial A$ | Result |
|---|---|---|---|
| $A = 0$, $B = 0$ | $0$ (because $Ax = 0$) | $0$ (because $B = 0$) | Nothing ever changes. Training is stuck. |
| $A$ random, $B = 0$ | not $0$ | $0$ at step 1 | $B$ changes at step 1. Then $B \neq 0$, so $A$ also starts to change. |

Thus one matrix must be zero (to start equal to the base model) and the other must be random (so that the gradients are not zero).

## 6. The scale $\alpha / r$

The factor $\alpha / r$ multiplies the LoRA output. We use $\alpha = 32$ and $r = 16$, so $\alpha / r = 2$. If you change $r$ and keep $\alpha / r$ the same, the size of the update stays approximately the same. Then you do not have to find a new learning rate.

## 7. Dropout

During training only, LoRA applies dropout with $p = 0.05$ to its input:

$$
y = W x + \frac{\alpha}{r}\, B A\, \operatorname{dropout}(x)
$$

## 8. Jevstral numbers

LoRA is on 7 module types in each of the 34 layers:

| Module | $d_{in} \to d_{out}$ | LoRA weights |
|---|---|---|
| `q_proj`, `o_proj` | $4096 \to 4096$ | 131,072 each |
| `k_proj`, `v_proj` | $4096 \to 1024$ | 81,920 each |
| `gate_proj`, `up_proj` | $4096 \to 14336$ | 294,912 each |
| `down_proj` | $14336 \to 4096$ | 294,912 |

One layer: 1,310,720. All 34 layers: **44,564,480**, approximately 0.5 % of the model.

All trainable weights:

| Part | Weights |
|---|---|
| LoRA | 44,564,480 |
| Pointer head | 2,097,664 |
| Delimiter embedding rows (section 9) | 20,480 |
| **Total** | **46,682,624** |

Memory, approximately:

| | Calculation | Memory |
|---|---|---|
| Frozen base weights (fp32) | $\approx 8 \times 10^9 \times 4$ B | $\approx 32$ GB |
| Trainable weights with gradients and AdamW | $46.7 \times 10^6 \times 16$ B | $\approx 0.75$ GB |
| Activations | depends on the row length | the rest of the 80 GB |

## 9. Trainable token rows

The five delimiter tokens get trainable embedding rows. PEFT (`trainable_token_indices`) keeps a trainable matrix $D$ with one row for each delimiter. In the forward pass, $D$ replaces these rows. It is not added to them:

$$
E'_i = D_i \quad \text{for } i \in \{20, 21, 22, 23, 26\}, \qquad E'_i = E_i \quad \text{for all other } i
$$

$D$ has $5 \times 4096 = 20{,}480$ numbers. All other rows of the embedding matrix $E$ stay frozen.

**Start values.** In Ministral 3, the five original rows are all zeros (measured by `inspect_data`). Zero rows are identical, and RMSNorm of a zero vector gives very large gradients. Thus stage 1 starts each row $D_i$ as a sample from a normal distribution with the mean $\mu$ and covariance $\Sigma$ of the trained rows of $E$:

$$
D_i \sim \mathcal{N}(\mu, \Sigma)
$$

The code does not build the $4096 \times 4096$ matrix $\Sigma$. Let $X$ be the $n$ trained rows. With $w \sim \mathcal{N}(0, I_n)$:

$$
D_i = \mu + \frac{1}{\sqrt{n}} (X - \mu)^\top w
$$

This has the covariance $\frac{1}{n}(X - \mu)^\top (X - \mu) = \Sigma$.

Hugging Face `resize_token_embeddings` uses the same distribution, but multiplies $\Sigma$ by $10^{-9}$. Then all new rows are almost equal to $\mu$. That method is for new words in a text generator. Jevstral needs five different delimiters, so it uses the full $\Sigma$.

## 10. Checkpoints

A checkpoint holds only $A$, $B$, $D$ and the head. It does not hold the base model. The size is approximately 190 MB in fp32.
