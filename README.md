# freerex
very adaptive optimizers in tensorflow

Three optimizers that provably achieve optimal convergence rates with no prior information about the data. 
FreeRexDiag is a diagonal updates optimizer (similar to AdaGrad)
FreeRexSphere uses an L2 update for dimension-independence (good for high-dimensional problems)
FreeRexLayerWise is an intermediate between the above two that might be a little faster.
