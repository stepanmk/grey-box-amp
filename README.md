# Neural Grey-Box Guitar Amplifier Modelling with Limited Data
This repository contains supplementary material for the DAFx23 paper [Neural Grey-Box Guitar Amplifier Modelling with Limited Data](/).

### Abstract
This paper combines recurrent neural networks (RNNs) with the discretised Kirchhoff nodal analysis (DK-method) to create a grey-box guitar amplifier model. Both the objective and subjective results suggest that the proposed model is able to outperform a baseline black-box RNN model in the task of modelling a guitar amplifier, including realistically recreating the behaviour of the amplifier equaliser circuit, whilst requiring significantly less training data. Furthermore, we adapt the linear part of the DK-method in a deep learning scenario to derive multiple state-space filters simultaneously. We frequency sample the filter transfer functions in parallel and perform frequency domain filtering to considerably reduce the required training times compared to recursive state-space filtering. This study shows that it is a powerful idea to separately model the linear and nonlinear parts of a guitar amplifier using supervised learning.

### Resources
- [Listening examples](https://stepanmk.github.io/grey-box-amp/)
- [Dataset](https://zenodo.org/record/7970723)
