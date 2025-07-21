import torch
from torch.quasirandom import SobolEngine

@torch.no_grad()
def quasi_monte_carlo_gumbel_samples(num_samples, logits_shape):
    """Generate quasi-Monte Carlo Gumbel samples using Sobol sequence."""
    # Use a Sobol engine to generate quasi-random numbers in [0, 1]
    print(logits_shape)
    steps, num_classes = logits_shape
    dimension=num_classes * steps
    sobol_engine = SobolEngine(dimension=dimension, scramble=True)
    U = sobol_engine.draw(num_samples)  # Quasi-random uniform samples in [0, 1]
    U=U.view(num_samples, steps, num_classes)
    # Convert uniform samples to Gumbel(0, 1) samples
    G = -torch.log(-torch.log(U))  # Inverse CDF (quantile function) of Gumbel
  
    return G

@torch.no_grad()
def conditional_gumbel(logits, D, k=1):
    """Outputs k samples of Q = Gumbel(0, 1) using quasi-Monte Carlo sampling."""
    print(logits.shape,"shape of logits @@@@@@@@333333333")
    # Generate quasi-Monte Carlo Gumbel samples
    E = quasi_monte_carlo_gumbel_samples(k, logits.shape)  # Shape: (k, num_classes) #logits shape [-1]
    print(D.shape,"shape of D is ...................")
    # Check for NaN values
    print(torch.isnan(E).any(),": E nan value")
    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True)  
    print(Ei.shape,"Shape of Ei")
    # Check for very small positive numbers
    very_small_positive = (E > 0) & (E < 1e-4)

    # Check if any very small positive numbers exist
    contains_very_small_positive = very_small_positive.any()
    print(contains_very_small_positive,"check for Ei")
    print(torch.isnan(Ei).any(),": Ei nan value")
    
    # Partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)  
    print(torch.isnan(Z).any(),": Z nan value")
   
    
    # Sampled Gumbel-adjusted logits
    # Need to change the formula for adjusted
    
    # adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
    #             (1 - D) * -torch.log(E / torch.exp(logits) + Ei / Z))
    
    adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
                (1 - D) * -torch.log(E / torch.exp(logits) + Ei / Z))
    
    print(torch.isnan((-torch.log(Ei))).any(),"adjusted first term")
    print(torch.isnan(-torch.log(E / torch.exp(logits) + Ei / Z)).any(),"adjusted second term")
    print(adjusted.shape,"99090909090900909090999999999999999999999999")
    
    return adjusted - logits


def exact_conditional_gumbel(logits, D, k=1):
    """Same as conditional_gumbel but uses rejection sampling."""
    # Rejection sampling.
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


def gumbel_rao(logits, k, temp=1.0, I=None):
    """Returns a categorical sample from logits (over axis=-1) as a
    one-hot vector, with Gumbel-Rao gradient.

    k: integer number of samples to use in the Rao-Blackwellization.
    1 sample reduces to straight-through Gumbel-Softmax.

    I: optional, categorical sample to use instead of drawing a new
    sample. Should be a tensor(shape=logits.shape[:-1], dtype=int64).

    """
    num_classes = logits.shape[-1]
    if I is None:
        I = torch.distributions.categorical.Categorical(logits=logits).sample()
    D = torch.nn.functional.one_hot(I, num_classes).float()
    adjusted = logits + conditional_gumbel(logits, D, k=k)
    surrogate = torch.nn.functional.softmax(adjusted/temp, dim=-1).mean(dim=0)
    
    return replace_gradient(D, surrogate)

# # # Example usage:
# logits = torch.randn(10,5 )  # Example logits for 5 samples, 10 classes
# k = 10
# output = gumbel_rao(logits, k, temp=1.0)
# print(output.shape)  # Ensure the output shape is as expected

if __name__ == '__main__':
    logits = torch.randn((5, 2), requires_grad=True)  # Random logits for 5 classes
    optimizer = torch.optim.SGD([logits], lr=0.1)  # Simple SGD optimizer
    k = 10  # Number of samples for QMC
    criterion = torch.nn.CrossEntropyLoss()  # Cross-entropy loss

    for epoch in range(10):
        optimizer.zero_grad()
        sample = gumbel_rao(logits, k)
        target = torch.argmax(sample).unsqueeze(0)  # Convert one-hot to class index
        loss = criterion(logits.unsqueeze(0), target)  # Compute cross-entropy loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}:")
        print("Logits:", logits)
        print("Sample:", sample)
        print("Loss:", loss.item())