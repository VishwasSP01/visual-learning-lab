# Our first neural network

## 1. An image is a collection of numbers

Each pixel has three RGB values, between 0 and 255:

- Blue: `(0, 0, 255)`.
- Red: `(255, 0, 0)`.
- Green in this dataset: `(0, 128, 0)`.
- White: `(255, 255, 255)`.

Before learning, we divide each value by 255. Blue becomes `(0, 0, 1)`.
One image has shape `(64, 64, 3)`: height, width, and color channels.
A batch adds another dimension: `(batch_size, 64, 64, 3)`.

## 2. The network predicts a new color

Our network is `3 -> 16 -> 3`: three input color numbers, sixteen hidden units,
then three predicted color numbers. ReLU is the activation between the layers.
The same network is applied independently to every pixel, sharing all weights.

There are 115 learned parameters: `(3 * 16 + 16) + (16 * 3 + 3)`.
The added 16 and 3 are biases. We do not encode a blue-to-red rule in this network.

It has no access to neighboring pixels, coordinates, masks, or instructions.
The network therefore learns a color mapping, not the concept of a circle.

## 3. Loss measures mistakes

The target image supplies the desired colors during training. Mean squared
error (MSE) measures the average squared difference between predicted and target
color values. We calculate it on normalized values, before rounding or clipping.

The first run's validation MSE fell from about 1.515 to about 1.4e-17.
This extremely small error is possible because the data contains only three
distinct input colors, repeated across many pixels and layouts.

## 4. Backpropagation tells us how to adjust the weights

The important lines in the training loop are:

```python
prediction = model(inputs)
loss = nn.functional.mse_loss(prediction, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`zero_grad()` clears the previous gradients. `backward()` computes derivatives
of the loss with respect to the parameters. `step()` uses those gradients to
update the parameters with Adam. Repeating this is how the network learns.

One epoch is one pass through the 5,000 training images. The first experiment
ran ten epochs and selected the checkpoint with the lowest validation MSE.
Validation does not update weights; it chooses among the saved candidates.

## 5. Save images and measure the result

To save a prediction, we clamp normalized RGB to `[0, 1]`, multiply by 255,
round, and convert to 8-bit pixels. The exact-pixel evaluation uses these saved
values. This conversion is declared in the report; it is distinct from raw MSE.

The selected model matched all target pixels in the 500 validation images.
The fixed color-replacement rule also scores perfectly on this task.
The 1,000 test images were not used to train or select this model.

This is an introductory neural-network experiment. It does not establish
spatial reasoning, understanding language, or editing real photographs.

See [PyTorch's optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
for the training-loop principles used here.
