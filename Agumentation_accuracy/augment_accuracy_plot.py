# Plot the training and test error as a function of the number of
# training examples on log-log scale
import numpy as np
import matplotlib.pyplot as plt

# Below are the loss values given in the cmd after execution
number_training = [(i+1) for i in range(10)]

train_accuracy_normal = [0.8496, 0.9132, 0.9229, 0.9306, 0.9353, 0.9380, 0.9408, 0.9409, 0.9426, 0.9423]
valid_accuracy_normal = [0.9473, 0.9582, 0.9631, 0.9665, 0.9678, 0.9680, 0.9695, 0.9697, 0.9702, 0.9709]

train_accuracy_augment = [0.6901, 0.8151, 0.8372, 0.8525, 0.8595, 0.8652, 0.8704, 0.8697, 0.8753, 0.8744]
valid_accuracy_augment = [0.9430, 0.9498, 0.9558, 0.9592, 0.9622, 0.9638, 0.9647, 0.9653, 0.9674, 0.9659]

# Plotting accuracy of training sets
plt.figure()
# Plot the data of training accuracies
plt.plot(number_training, train_accuracy_normal)
plt.plot(number_training, train_accuracy_augment)
# Label the axes, legends and title
plt.xlabel('Epoches')
plt.ylabel('Training Accuracy')
plt.legend(['Normal', 'Augmented'], loc = 'best')
plt.title('Effect of augmentation on training accuracy')
# Show and save the figure we obtained
plt.savefig('Training_accuracy_augment')
plt.show()

# Plotting accuracy of training sets
plt.figure()
# Plot the data of training accuracies
plt.plot(number_training, valid_accuracy_normal)
plt.plot(number_training, valid_accuracy_augment)
# Label the axes, legends and title
plt.xlabel('Epoches')
plt.ylabel('Validation Accuracy')
plt.legend(['Normal', 'Augmented'], loc = 'best')
plt.title('Effect of augmentation on validating accuracy')
# Show and save the figure we obtained
plt.savefig('Validation_accuracy_augment')
plt.show()
