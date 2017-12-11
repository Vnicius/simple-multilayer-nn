# simple-multilayer-nn
A simple multilayer Neural Network with TensorFlow.

The code get the dataset mnist of TensorFlow to train a Neural Network with dimensions defined by parameters.

*Based in [this](https://www.youtube.com/watch?v=BhpvH5DuVu8) tutorial video*

## Dependences

- Python = 3.x
- TensorFlow = 1.4

## Run

```console
    $ python3 nn.py [1] [2] [3] [4]
```

1. Number of layers
2. Number of nodes of each layer
3. Number of epochs of train
4. The size of batch to train

## Examples

**Running:**
```console
    $ python3 nn.py 3 500 10 100
```

**Expected Output:**
```console
    Extracting /tmp/data/train-images-idx3-ubyte.gz
    Extracting /tmp/data/train-labels-idx1-ubyte.gz
    Extracting /tmp/data/t10k-images-idx3-ubyte.gz
    Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
    
    Epoch: 1 of 10
    Loss: 173283.258811

    Epoch: 2 of 10
    Loss: 37558.9933906

    Epoch: 3 of 10
    Loss: 22394.4671686

    Epoch: 4 of 10
    Loss: 14816.13091

    Epoch: 5 of 10
    Loss: 9734.38732906

    Epoch: 6 of 10
    Loss: 6604.00602329

    Epoch: 7 of 10
    Loss: 4680.07977169

    Epoch: 8 of 10
    Loss: 3127.89936074

    Epoch: 9 of 10
    Loss: 2279.50701406

    Epoch: 10 of 10
    Loss: 1824.41436973

    Accuracy: 95.4900026321%
```