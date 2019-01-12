# Implementations of various machine learning papers

## Setup

Models are built in python (3.6) using [TensorFlow](https://www.tensorflow.org/).

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

Train and evaluate a model via run script, e.g.

```bash
python run.py --model=vgg
```

Monitor training by launching TensorBoard local server pointed at the model's logs directory, e.g.
```bash
tensorboard --logdir=vgg/logs
```

## Papers

1. VGG: Image classification with deep convolutional network. Implements VGG architecture from
[Very Deep Convolutional Networks for Large-Scale Image Recognition, Simonyan and Zisserman 2015](https://arxiv.org/pdf/1409.1556.pdf)

2. NMT: Encoder-Decoder architecture for natural language translation [Neural Machine Translation by Jointly Learning to Align and Translate
](https://arxiv.org/abs/1409.0473)

3. Pixel-level scene segmentation with a multi-scale convolutional network. Implements architecture from [Learning Hierarchical Features for Scene Labeling](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
