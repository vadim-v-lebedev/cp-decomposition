This is an implemetation of the method from our paper **Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition**
https://arxiv.org/pdf/1412.6553.pdf
MNIST example, based on MNIST example from caffe, is provided.

Requrements: caffe, python with numpy and scikit-tensor

How to make it work:

1. Set paths to your caffe installation in **paths.py**
2. In lenet/lenet.prototxt, edit "source" params of input layers, or copy **mnist_train_lmbd** and **mnist_test_lmdb** from caffe/examples/mnist here. LeNet needs input data!
3. run lenet/main.py, for example like this `python lenet/main.py 5 conv2`. First parameter of this script is the number of components R, and the second is layer name. Biggger R leads to more accurate, but slower models. The script will produce model lenet_accelerated.prototxt and weights file lenet_accelerated.caffemodel
4. Now you can evaluate accelerated model ```
$CAFFE_ROOT/build/tools/caffe time --model lenet_accelerated.prototxt
$CAFFE_ROOT/build/tools/caffe test --model lenet_accelerated.prototxt -weights lenet_accelerated.caffemodel```
5. As shown in the paper, finetuning of accelerated model can improve accuracy
