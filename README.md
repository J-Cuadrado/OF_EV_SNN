# Optical Flow estimation from Event Cameras and Spiknig Neural Networks

We present here the code we have used to develop our Spiking Neural Network (SNN), capable of accurately estimating optical flow from event data. Our main contributions are:

- 3d encoders for temporal dependencies
- novel loss function
- stateless snn
- accurate results on DSEC

We have used the DSEC Dataset () to train and evaluate our models, and we report our performance on their official optical flow benchmark.

Our network has been developped with PyTorch (https://pytorch.org/), and we have used the SpikingJelly library (https://github.com/fangwei123456/spikingjelly) for our spiking neuron model. We have used the SpikingJelly version 0.0.0.0.13.

Below can be found an example of the results obtained with our model, evaluated on our validation set. Each quadrant represents:
- Upper-left: Optical Flow ground-truth.
- Upper-right: Masked prediction (only valid pixels are shown)
- Lower-left: optical flow encoding. The estimation at each time step is represented as a Lab image, where the a and b channels represent the x- and y- components of the optical flow, and the L channel accounts for the optical flow magnitude.
- Lower-right: Unmasked prediction. It can be seen that, even if the network was not explicitly trained on the task, it is nonetheless able to obtain a general comprehension of the visual scene, and to show contours (e.g. trafic signals)

![ezgif-2-ec655c32b9](https://user-images.githubusercontent.com/71754039/217257601-344dc0f9-f58c-4981-b9b1-21b46b35751e.gif)


## References

WIP
