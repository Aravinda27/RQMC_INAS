#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 22:03:24 2024

@author: user1
"""
import matplotlib.pyplot as plt
import numpy as np

# Generate some example data (replace with actual data)
epochs = np.arange(0, 101)
train_loss_softmax = np.random.uniform(0.1, 0.5, size=len(epochs))
val_loss_softmax = np.random.uniform(0.1, 0.5, size=len(epochs))
train_loss_gumbel = np.random.uniform(0.1, 0.5, size=len(epochs))
val_loss_gumbel = np.random.uniform(0.1, 0.5, size=len(epochs))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_softmax, label="Train Loss (Softmax)", color="blue")
plt.plot(epochs, val_loss_softmax, label="Validation Loss (Softmax)", color="red")
plt.plot(epochs, train_loss_gumbel, label="Train Loss (Gumbel M=15 λ=10)", color="purple")
plt.plot(epochs, val_loss_gumbel, label="Validation Loss (Gumbel M=15 λ=10)", color="orange")

# Add labels and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses over Epochs")
plt.grid(True)
plt.legend()

# Set axis limits
plt.xlim(0, 100)
plt.ylim(0, 1)

# Save the plot as an image
plt.savefig("loss_graph.png")

# Show the plot
plt.show()

print("The loss graph has been saved as loss_graph.png")

