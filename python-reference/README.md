# Reference Python Implementation

This directory contains a reference digit classifier using a multi-layer perceptron neural network. This is built using only python and numpy. Matplotlib and ipykernel are used for visualization and notebook support respectively. Scikit learn is used to import mnist-784 data.

Tested using python 3.12.3 and numpy 2.1.1

Setting up an environment.

1. create a virtual environment
2. install requirements
3. register the ipython kernel with the local jupyter server.

```bash
python -m venv .venv

.venv/Scripts/activate

pip install -r requirements.txt

python -m ipykernel install --name digit_classifier
```

The appropriate kernel will be available from a local jupyter lab server or vs code.