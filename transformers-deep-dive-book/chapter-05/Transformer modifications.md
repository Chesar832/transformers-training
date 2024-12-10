# Transformer Modifications


## Transformer block modifications

### Lightweight Transformers
#### Funnel-transformer
This model is a transformation of the original Tranformer architecture which takes the ouput of each layer and apply pooling before passes it to the next layer. This modification reduces the computational cost and allows to add nire kayers to support deeper models.

So, if the output of a given layer is $h$ and the output of the pooling layer for that layer is $h' = Pooling(h)$ where $h \in \mathbb{R}^{T \times d}$ and $h' \in \mathbb{R}^{T' \times d}$, for some $T' < T$.

Before continuing, let's break the previous expressions to really really understand what is happening in a mathematical point of view.

So:
- s
 

---

## Transformers with modified multi-head self-attention



---

## Modifications for training task efficiency




---

# Transformer submodule changes





