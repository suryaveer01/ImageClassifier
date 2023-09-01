import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Number of training epochs
training_loss = [0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06]  # Training loss
validation_loss = [1.5, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.37, 0.36]  # Validation loss

# Plot the generalization error curve
plt.plot(epochs, training_loss, 'b', label='Training Loss')
plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
plt.title('Generalization Error Curve')
plt.xlabel('Number of Training Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()