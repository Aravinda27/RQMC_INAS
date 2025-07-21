import torch
from torch.quasirandom import SobolEngine

@torch.no_grad()
def conditional_gumbel_qmc(logits, D, k=1000):
    """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
    + Q) is given by D (one hot vector) using Quasi-Monte Carlo sampling."""
    # Sobol sequence generator
    sobol = SobolEngine(dimension=logits.shape[-1], scramble=True)
    # Generate Sobol samples
    U = sobol.draw(k)
    # Convert uniform samples to Gumbel samples
    G = -torch.log(-torch.log(U))
    # Expand D to match the number of QMC samples
    D_expanded = D.unsqueeze(0).expand(k, -1, -1)
    # G of the chosen class
    Gi = (D_expanded * G.unsqueeze(1)).sum(dim=-1, keepdim=True)
    # Partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    # Sampled gumbel-adjusted logits
    adjusted = (D_expanded * (Gi + torch.log(Z)) +
                (1 - D_expanded) * (G.unsqueeze(1) + torch.log(Z) - torch.log(logits.exp() + Gi.exp())))
    return adjusted - logits

def exact_conditional_gumbel(logits, D, k=1):
    """Same as conditional_gumbel but uses rejection sampling."""
    idx = D.argmax(dim=-1)
    gumbels = []
    while len(gumbels) < k:
        gumbel = torch.rand_like(logits).log().neg().log().neg()
        if logits.add(gumbel).argmax() == idx:
            gumbels.append(gumbel)
    return torch.stack(gumbels)

def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return surrogate + (value - surrogate).detach()

def gumbel_rao(logits, k, temp=0.1, I=None):
    """Returns a categorical sample from logits (over axis=-1) as a
    one-hot vector, with gumbel-rao gradient.

    k: integer number of samples to use in the rao-blackwellization.
    1 sample reduces to straight-through gumbel-softmax.

    I: optional, categorical sample to use instead of drawing a new
    sample. Should be a tensor(shape=logits.shape[:-1], dtype=int64).

    """
    num_classes = logits.shape[-1]
    if I is None:
        I = torch.distributions.categorical.Categorical(logits=logits).sample()
    D = torch.nn.functional.one_hot(I, num_classes).float()
    adjusted = logits + conditional_gumbel_qmc(logits, D, k=k)
    surrogate = torch.nn.functional.softmax(adjusted/temp, dim=-1).mean(dim=0)
    return replace_gradient(D, surrogate)

# Example usage
logits = torch.randn((17, 2), requires_grad=True)  # Random logits for 5 samples, each with 2 classes
optimizer = torch.optim.SGD([logits], lr=0.1)  # Simple SGD optimizer
k = 1000  # Number of samples for QMC
criterion = torch.nn.CrossEntropyLoss()  # Cross-entropy loss

for epoch in range(10):
    optimizer.zero_grad()
    sample = gumbel_rao(logits, k)
    target = torch.argmax(sample, dim=-1)  # Convert one-hot to class index
    loss = criterion(logits, target)  # Compute cross-entropy loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}:")
    print("Logits:", logits)
    print("Sample:", sample)
    print("Loss:", loss.item())
