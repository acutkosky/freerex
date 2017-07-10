# freerex
very adaptive optimizers in tensorflow based on <a href=http://proceedings.mlr.press/v65/cutkosky17a.html>this paper</a>.

Three optimizers that provably achieve optimal convergence rates with no prior information about the data.

`FreeRexDiag` is a coordinate-wise optimizer (this is probably the best default algorithm).

`FreeRexSphere` uses an L2 update for dimension-independence (good for high-dimensional problems).

`FreeRexLayerWise` is an intermediate between the above two that might be computationally faster than FreeRexSphere.

These are all implemented as subclasses of Tensorflow's optimizer class. You should be able to use them as drop-in replacements for other optimizers. For example:
```
optimizer = tf.Train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)
```
Can be replaced with
```
optimizer = FreeRex() # FreeRex is an alias for FreeRexDiag
train_step = optimizer.minimize(loss)
```

Each algorithm takes as input the parameter `k_inv` (e.g. `optimizer = FreeRex(0.5)`). This parameter is analagous to a learning rate, but provably requires less tuning. The default is `k_inv=1.0`, which has worked well in my limited experiments.
