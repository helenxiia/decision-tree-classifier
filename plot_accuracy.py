import matplotlib.pyplot as plt
test_weighted_acc = []
train_weighted_acc = []
test_unweighted_acc = []
train_unweighted_acc = []

with open ("test_weighted_classifier_accuracy2.txt") as file:
    for row in file:
        test_weighted_acc.append(float(row.strip()))

with open ("train_weighted_classifier_accuracy2.txt") as file:
    for row in file:
        train_weighted_acc.append(float(row.strip()))

with open ("test_unweighted_classifier_accuracy2.txt") as file:
    for row in file:
        test_unweighted_acc.append(float(row.strip()))

with open ("train_unweighted_classifier_accuracy2.txt") as file:
    for row in file:
        train_unweighted_acc.append(float(row.strip()))

plt.plot(test_weighted_acc, label='Test Data Weighted Accuracy')
plt.plot(train_weighted_acc, label='Train Data Weighted Accuracy')
plt.xlabel('Number of internal nodes')
plt.ylabel('% Accuracy')
plt.title('Method 2: Weighted Classifier Accuracy')
plt.legend()
plt.show()
print("shown 1")
print(test_weighted_acc)
print(train_weighted_acc)
plt.savefig('weighted_accuracy_plot.png') 
plt.close()

plt.plot(test_unweighted_acc, label='Test Data Unweighted Accuracy')
plt.plot(train_unweighted_acc, label='Train Data Unweighted Accuracy')
plt.xlabel('Number of internal nodes')
plt.ylabel('% Accuracy')
plt.title('Method 1: Unweighted Classifier Accuracy')
plt.legend()
plt.show()
print("shown 2")
print(test_unweighted_acc)
print(train_unweighted_acc)
plt.savefig('unweighted_accuracy_plot.png') 

