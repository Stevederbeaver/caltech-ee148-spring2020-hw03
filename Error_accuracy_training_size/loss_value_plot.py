# Plot the training and test error as a function of the number of
# training examples on log-log scale
import numpy as np
import matplotlib.pyplot as plt

# Below are the loss values given in the cmd after execution
number_training = [(i / 16.0) * 60000.0 for i in [1, 2, 4, 8, 12, 16]]

# Epoch = 10
train_loss_list = [0.0621, 0.0371, 0.0254, 0.0182, 0.0171, 0.0142]
test_loss_list = [0.0464, 0.0271, 0.0191, 0.0170, 0.0176, 0.0181]

train_accuracy_list = [0.9814, 0.9888, 0.9922, 0.9946, 0.9952, 0.9958]
test_accuracy_list = [0.9844, 0.9914, 0.9940, 0.9937, 0.9945, 0.9946]

# Plotting testing errors of two datasets
plt.figure()
# Set log scale for the X, Y axes
ax_2 = plt.gca()
ax_2.set_xscale("log")
ax_2.set_yscale('log')
# Plot the data of training errors we obtained above
plt.plot(number_training, train_loss_list)
plt.plot(number_training, test_loss_list)
# Label the axes, legends and title
plt.xlabel('Number of training samples')
plt.ylabel('Average Error')
plt.legend(['training set', 'testing set'], loc = 'best')
plt.title('Error versus number of training samples')
# Show and save the figure we obtained
plt.savefig('Error_training_samples')
plt.show()

# Plotting Eulidean Norms of two datasets' weight vectors
plt.figure()
# Plot the data of training errors we obtained above
plt.plot(number_training, train_accuracy_list)
plt.plot(number_training, test_accuracy_list)
# Label the axes, legends and title
plt.xlabel('Number of testing samples')
plt.ylabel('Average Accuracy')
plt.legend(['training set', 'testing set'], loc = 'best')
plt.title('Accuracy versus number of training samples')
# Show and save the figure we obtained
plt.savefig('Accuracy_training_samples')
plt.show()
