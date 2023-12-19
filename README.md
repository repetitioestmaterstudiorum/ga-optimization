# ga-optimization

Training a simple artificial neural network (ANN) using a genetic algorithm (GA). This means to optimize the weights and biases of the ann using a GA.

I used PyTorch as the deep learning framework, and implemented the GA myself.

## What is a Genetic Algorithm?

Pseudocode for a simple GA:

```
initialize population
while not stop condition:
    evaluate population
    select parents
    crossover parents
    mutate offspring
    replace population with offspring
```

There are many variations of the GA, e.g. different selection methods, different crossover methods, different mutation methods, etc. The pseudocode above is a very simple GA.

GAs are inspired by the process of natural selection in evolution. Their performance is usually much better than brute force algorithms, but they are not guaranteed to find the optimal solution, and usually perform worse than problem-specific algorithms. Still, they are very valuable general-purpose optimization algorithms, especially for problems that do not have a known solution.

What I find most interesting about GAs is their inspiration by nature. The idea of using a GA to train an ANN is especially interesting, because it is a combination of two different ideas inspired by nature (evolution and the brain).

## Data

### XOR Gate

The XOR gate is a simple logic gate with two inputs (A and B) and one output. The output is `1` if one of the inputs is `1` and the other is `0`.

|  A  |  B  | AND | OR  | XOR |
| :-: | :-: | :-: | :-: | :-: |
|  1  |  1  |  1  |  1  |  0  |
|  1  |  0  |  0  |  1  |  1  |
|  0  |  1  |  0  |  1  |  1  |
|  0  |  0  |  0  |  0  |  0  |

### IRIS Dataset

The [IRIS dataset](https://archive.ics.uci.edu/ml/datasets/iris) is small, multivariate dataset with 150 samples. Each sample has four features (sepal length, sepal width, petal length, petal width) and a label (Iris Setosa, Iris Versicolour, Iris Virginica).

## Results

### XOR Gate (implemented as a regression problem)

**Genetic Algorithm**

The best trial according to the amount of epochs required to achieve a loss below the loss threshold (0.00001) took 64 epochs. Training time was 5.80 seconds.

**Gradient Descent (PyTorch SGD optimizer)**

After manual tuning, with a learning rate of 0.1, momentum of 0.9 and weight decay of 0.0001, the training time remains constant, but only approximately 150 epochs are required.
It’s notable, however, that gradient descent does not converge every approximately 5th time with optimized hyperparameters.

**Conclusion**

The tuned genetic algorithm is more reliable than the tuned gradient descent algorithm in solving the XOR problem with the desired network architecture, in that it converges more frequently. On the other hand, the gradient descent algorithm is much more time efficient for this problem.

### IRIS Dataset (classification problem)

**Genetic Algorithm**

Accuracy: 1.0 on validation data, and 0.9886 on training data.
Loss: 0.0235 on validation data, and 0.0540 on training data.

**Gradient Descent (PyTorch SGD optimizer)**

Accuracy: 1.0 on validation data, and 0.9772 on training data.
Loss: 0.0450 on validation data, and 0.0593 on training data.

**Conclusion**

The two algorithms achieved the same accuracy for the validation set, but the genetic algorithm achieved a slightly higher accuracy on the training dataset.

Also, the loss was lower for the genetic algorithm for both datasets. This is especially interesting considering the GD’s early stopping criteria that checks the last 2000 losses for a standard deviation below 1e-6 on both datasets, which came into effect, meaning that it is quite unlikely that the algorithm would have achieved a lower loss if it had kept training, and that was in combination with a learning rate scheduler with a patience of 10.

In terms of time complexity, the GA was an order of magnitude slower than GD on the XOR dataset, and two orders of magnitude on the IRIS dataset (without an early stopping criteria that allows for less than 100% accuracy on both datasets).

## Limitations and Possible Future Improvements

The experiments have a number of limitations:

- The SGD algorithm in PyTorch is highly optimized and has been developed in many
  iterations, whereas the GA was implemented by myself within a short period of time
- The GA could be optimized by implementing more variations of selection, crossover,
  mutation, and other functionality
- More parameter control logic could be implemented for the GA (The learning curves
  for the GA look rough, which could indicate a lack of parameter control. Also, introducing one controlled parameter for the mutation ranges improved the results immensely, which is a promising sign for more of the same)
- A higher number of trials per experiment would improve the representativeness of the findings
- The early stopping criteria could be fine-tuned more precisely depending on the dataset
- Both datasets were relatively small (4 and 150 examples). The comparison could look quite differently on larger datasets and larger ANNs
