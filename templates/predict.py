import torch

# A dummy tensor with a shape that simulates (batch_size, feature_size)
# For simplicity, let's say our feature_size is 6 (simulating 2 heads, each with 3 features for Q, K, V)
tensor = torch.arange(24).reshape(4, 6)  # Shape is [4, 6]

# Reshape the tensor to simulate (batch_size, num_heads, features_per_head)
# Here, let's assume 2 heads, so features_per_head will be 3
tensor_reshaped = tensor.reshape(4, 2, 3)  # Shape is now [4, 2, 3]

# Permute the dimensions to change the order
tensor_permuted = tensor_reshaped.permute(0, 2, 1)  # Shape is now [4, 3, 2]

print("Original shape:", tensor)
print("Reshaped shape:", tensor_reshaped)
print("Permuted shape:", tensor_permuted)