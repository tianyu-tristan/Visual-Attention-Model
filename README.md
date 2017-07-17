# Visual Attention Model

The main idea of this exercise is to study the evolvement of the state of the art and main work along topic of visual attention model. There are two datasets that are studied: augmented MNIST and SVHN. The former dataset focused on canonical problem  —  handwritten digits recognition, but with cluttering and translation, the latter focus on real world problem  —  street view house number (SVHN) transcription. In this exercise, the following papers are studied in the way of developing a good intuition to choose a proper model to tackle each of the above challenges.

For augmented MNIST dataset:
* The baseline model is based on classical 2 layer CNN
* The target model is recurrent attention model (RAM) with LSTM, refer to paper [2]

For SVHN dataset:
* The baseline model is based on 11 layer CNN: with convolutional network to extract image feature, then use multiple independent dense layer to predict ordered sequence, refer to paper [1]
* The target model is deep recurrent attention model (DRAM) with LSTM and convolutional network, refer to paper [3]

Additionally:
* Spatial Transformer Network is also studied as latest development in the visual attention regime, refer to paper [5]

Both of the above dataset challenges focuses on digit recognition. In this exercise, MNIST is used to demonstrate the solution for single digit recognition, whereas SVHN is used to show the result of multiple digit sequence recognition.


For more detail, please refer to [this blog](https://medium.com/@tianyu.tristan/visual-attention-model-in-deep-learning-708813c2912c)


[1] Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks. Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay Shet (2013). https://arxiv.org/abs/1312.6082

[2] Recurrent Models of Visual Attention, Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu, https://arxiv.org/abs/1406.6247

[3] Multiple Object Recognition with Visual Attention, Jimmy Ba, Volodymyr Mnih, Koray Kavukcuoglu, https://arxiv.org/abs/1412.7755

[4] Reading Digits in Natural Images with Unsupervised Feature Learning , Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng on NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf

[5] Spatial Transformer Networks, Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu, https://arxiv.org/abs/1506.02025

[6] Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, Williams et al, http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
