In this assignment you will practice writing backpropagation code, and training
Neural Networks. The goals of this assignment are as follows:

- develop proficiency in writing efficient vectorized code with numpy
- understand **Neural Networks** and how they are arranged in layered
  architectures
- understand and be able to implement (vectorized) **backpropagation**
- implement various **update rules** used to optimize Neural Networks
- implement **Dropout** to regularize networks
- implement **Batch Normalization** and **Layer Normalization** for training deep networks
- understand and implement both Vanilla RNNs and Long-Short Term Memory (LSTM) networks


## Setup
Get the code as a zip file at the channel.

You can follow the setup instructions [here](http://cs231n.github.io/setup-instructions/). This is Stanford cs231n tutorial, unfortunately in this course you won't have extra $100 for your Google Cloud, but you still will have $300 free credit for sign up.

### Download data:
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the `backpropagation_workshop` directory:

```bash
cd cs231n/datasets
./get_datasets.sh
```

### Start IPython:
After you have the CIFAR-10 data, you should start the IPython notebook server from the
`backpropagation_workshop` directory, with the `jupyter notebook` command. (See the [Google Cloud Tutorial](http://cs231n.github.io/gce-tutorial/) for any additional steps you may need to do for setting this up, if you are working remotely).

### Some Notes
**NOTE:** This year, the `backpropagation_workshop` code has been tested to be compatible with python version `3.6` (it may work with other versions of `3.x`, but we won't be officially supporting them). You will need to make sure that during your virtual environment setup that the correct version of `python` is used. You can confirm your python version by (1) activating your virtualenv and (2) running `which python`.

### Task 1: Softmax
The IPython Notebook softmax.ipynb will walk you through implementing linear classifier.

### Task 2: Two-Layer Neural Network
The IPython Notebook two_layer_net.ipynb will walk you through the implementation of a two-layer neural network classifier

### Task 3: Fully-connected Neural Network
The IPython notebook `task3_FullyConnectedNets.ipynb` will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

### Task 4: Dropout
The IPython notebook `task4_Dropout.ipynb` will help you implement Dropout and explore
its effects on model generalization.

### Task 5: Image Captioning with Vanilla RNNs
The Jupyter notebook `task5_RNN_Captioning.ipynb` will walk you through the implementation of
an image captioning system on MS-COCO using vanilla recurrent networks.

### Task 6: Image Captioning with LSTMs
The Jupyter notebook LSTM_Captioning.ipynb will walk you through the implementation of
Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

### Task 7: Layer Normalization
This task is fully optional and has no deadline.

In the IPython notebook `task7_LayerNormalization.ipynb` you will implement batch
normalization and layer normalization, and use them to train deep fully-connected networks.

### Submitting your work and grade others

Submit your work via Telegram @peer_review_bot

Example of task submission command:
```/send_task 2.1```

Where 2.1 is workshop_number.task_number.

After that, drop **one** task file. In the case of this workshop, zip and send one notebook (corresponding to the task number)
and cs231n directory with necessary for the task files.

After you submit the task, you need to check at least two other solutions. To get them use command
```/get_task 2.1 @username```

Where @username is telegram username of one you should grade.

After that, you should contact @username and to check the task in person (on in a call). To send your grade, use the command
```/grade 2.1 @username 8```

Where 8 is a scaled [0, 10] grade of that task.

**Important:** _Please make sure that the submitted notebooks have been run and the cell outputs are visible._


Credit: [cs231n.stanford.edu](http://cs231n.stanford.edu)
