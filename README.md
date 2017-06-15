# SVHN-paper-impl

RNN steering CNN for recognizing house number from google street view images

# Abstract

The idea of this final project is to efficiently detect house numbers from google street view images. The problem itself is similar to MNIST handwritten digits recognition, with additional challenges of digits sequencing and orientation.

Google has published paper with the idea to use RNN (LSTM) to steer CNN filter moving within house number digits detection. The main goal of this final project is to implement this idea, expecting to reuse this idea in other field in the future (e.g. genome analysis)

# Data

Google Street View House Number images (SVHN), available from here: http://ufldl.stanford.edu/housenumbers/

* 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
* 73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data

# Reference

[1] Multiple Object Recognition with Visual Attention, Jimmy Ba, Volodymyr Mnih, Koray Kavukcuoglu, https://arxiv.org/abs/1412.7755

[2] Recurrent Models of Visual Attention, Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu, https://arxiv.org/abs/1406.6247

[3] Reading Digits in Natural Images with Unsupervised Feature Learning , Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng on NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf
