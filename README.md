# Decision tree classifier

A decision tree classifier is used to classify documents into r/books and r/atheism.

Given words in a document, we build a decision tree using trainData.txt using maximum information gain.

Two methods are available to use for calculating the information gain.

1. Average information gain (evenly weighted across the leaves)
   
- $IG = I(E) - \left[ \frac{1}{2} \cdot I(E1) + \frac{1}{2} \cdot I(E2) \right]$

2. Information gain weighted by the fraction of documents on each side of the split

- $IG = I(E) - \left[ \frac{N1}{N} \cdot I(E1) + \frac{N2}{N} \cdot I(E2) \right]$

**Accuracy Charts**

![Unweighted Accuracy Plot](https://github.com/helenxiia/decision-tree-classifier/blob/main/unweighted_accuracy_plot.png)

![Weighted Accuracy Plot](https://github.com/helenxiia/decision-tree-classifier/blob/main/weighted_accuracy_plot.png)
