# PE bias

Experiments investigating the **positive evidence (PE) bias** in confidence judgements
of neural networks.

This code is based on
[taylorwwebb/performance_optimized_NN_confidence](https://github.com/taylorwwebb/performance_optimized_NN_confidence)
(code for *"Natural Statistics Support a Rational Account of Confidence Biases"*).

> **Heads up:** this repo is a work in progress.
> It will be cleaned up and refactored soon.

## Where the experiments live

All of the experiments that have actually been run are under **`v1/MNIST`**.
Other directories (`v1/RL`, `v1/CIFAR10`, `v2/...`) are not the focus here.

## Training and evaluation

Training and evaluation of the models is driven by scripts named `train_and_eval*.sh`
(or the underlying `train_and_eval*.py`). For example, from `v1/MNIST`:

```bash
./train_and_eval_3_fast.sh
```

Each `.sh` script is a thin wrapper that sweeps over runs / number of classes and calls the
matching `.py` file (e.g. `train_and_eval_3_fast.py`). You can also call the Python script
directly to control the arguments:

```bash
python3 ./train_and_eval_3_fast.py --run 1 --n_classes 10
```

## Testing / analysis

Testing and analysis scripts live under the various `test*/` folders
(`v1/MNIST/test/`, `v1/MNIST/test_fast/`, ...) and are named `PE_test*.sh`
(or the underlying `PE_test*.py`). For example:

```bash
cd v1/MNIST/test
./PE_test_3.sh
```

which sweeps over class counts and calls `PE_test_3.py`:

```bash
python3 ./PE_test_3.py --N_runs 25 --n_classes 10
```

## Environment setup (.venv)

The project targets **Python 3.9**. To reproduce the environment with a local virtual
environment:

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch notes:

- On macOS the default `torch` / `torchvision` wheels include MPS (Apple Silicon GPU) support.
- On Linux/Windows with a CUDA GPU, install the matching wheels following the selector at
  [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) instead of the
  generic ones pinned in `requirements.txt`.

To deactivate the environment when you're done:

```bash
deactivate
```
