import torch

# Create a 2D tensor
tensor = torch.tensor([[1, 2], [3, 4]])
print("Original Tensor:")
print(tensor)

# Create a tensor for indices
indices = torch.tensor([[0, 0], [1, 1]])
print("\nIndices:")
print(indices)

# Use gather to select elements
gathered_tensor = tensor.gather(1, indices)
print("\nGathered Tensor:")
print(gathered_tensor)
