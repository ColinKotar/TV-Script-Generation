# TV-Script-Generation

A Recurrent Neural Network (RNN) with Long Short Term Memory (LSTM) cells produces a made up TV script for an upcoming "episode". It learns the structure of how to write a TV script as well as how lines are spoken in broken English but, sometimes in perfect English. The dataset used is The Simpsons, but only a small portion of the 600 episodes available on Kaggle.

## Getting Started

Simply run the Jupyter Notebook dlnd_tv_script_generation.ipynb or you can run the script tv_script_generation.py

```
python tv_script_generation.py
```

### Prerequisites

You can install the required packages through Anaconda's environment manager using the machine-learning.yml file

```
conda env create -f machine-learning.yml
```

Then, activate the environment and run tv_script_generation.py

```
activate machine-learning
```

Otherwise, check out the machine-learning.yml file for dependencies and their versions

## Running the tests

Simply add test cases to problem_unittests.py or run it

```
python problem_unittests.py
```

## Built With

* [TensorFlow](https://www.tensorflow.org/install/install_windows) - The machine learning framework
* [Anaconda](https://repo.continuum.io/archive/Anaconda3-5.1.0-Windows-x86_64.exe) - The environment manager
* [Jupyter Notebook](http://jupyter.org/install) - The code documentation
