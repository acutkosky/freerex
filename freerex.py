'''
FreeRex optimizer
'''

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf
import numpy as np

from aggregate_optimizers import OptimizerWithAggregates

class FreeRexDiag(Optimizer):
    '''diagonal FreeExp Learner (does coordinate-wise updates ala adagrad'''
    def __init__(self, k_inv=1.0, log_scaling=True, set_scaling=None, epsilon=1e-6, use_locking=False, name='FreeRexDiag'):
        '''
        constructs a new freerex optimizer
        '''
        super(FreeRexDiag, self).__init__(use_locking, name)
        self._epsilon = epsilon
        self._k_inv = k_inv
        self._set_scaling = set_scaling
        self._log_scaling = log_scaling


    def _create_slots(self, var_list):
        total_length = np.sum([v.get_shape().num_elements() for v in var_list])
        for v in var_list:
            with ops.colocate_with(v):
                gradients_sum = constant_op.constant(0, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)
                grad_norm_sum = constant_op.constant(self._epsilon, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)
                absolute_regret = constant_op.constant(0, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)

                L = constant_op.constant(self._epsilon,
                                         shape=v.get_shape(),
                                         dtype=v.dtype.base_dtype)


                inverse_eta_squared = constant_op.constant(self._epsilon**2,
                                                           shape=v.get_shape(),
                                                           dtype=v.dtype.base_dtype)

                log_scaling = constant_op.constant(1.0,
                                                   shape=v.get_shape(),
                                                   dtype=v.dtype.base_dtype)
                if(self._set_scaling is not None):
                    scalings = constant_op.constant(self._set_scaling)
                else:
                    scalings = constant_op.constant(1.0)
                max_grad_norm = constant_op.constant(0.0)

                offset = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)

            self._get_or_make_slot(v, grad_norm_sum, "grad_norm_sum", self._name)
            self._get_or_make_slot(v, absolute_regret, "absolute_regret", self._name)
            self._get_or_make_slot(v, gradients_sum, "gradients_sum", self._name)
            self._get_or_make_slot(v, L, "L", self._name)
            self._get_or_make_slot(v, inverse_eta_squared, "inverse_eta_squared", self._name)
            self._get_or_make_slot(v, scalings, "scalings", self._name)
            self._get_or_make_slot(v, log_scaling, "log_scaling", self._name)
            self._get_or_make_slot(v, offset, "offset", self._name)
            self._get_or_make_slot(v, max_grad_norm, "max_grad_norm", self._name)

    def _apply_dense(self, grad, var):
        gradients_sum = self.get_slot(var, "gradients_sum")
        L = self.get_slot(var, "L")
        inverse_eta_squared = self.get_slot(var, "inverse_eta_squared")
        scalings = self.get_slot(var, "scalings")
        log_scaling = self.get_slot(var, "log_scaling")
        offset = self.get_slot(var, "offset")
        grad_norm_sum = self.get_slot(var, "grad_norm_sum")
        absolute_regret = self.get_slot(var, "absolute_regret")
        max_grad_norm = self.get_slot(var, "max_grad_norm")


        gradients_sum_update = gradients_sum + grad
        L_update = tf.maximum(L, tf.abs(grad))

        max_grad_norm_update = tf.maximum(max_grad_norm, tf.norm(grad))

        inverse_eta_squared_update = tf.maximum(inverse_eta_squared + 2*tf.square(grad), 
                                                L_update * tf.abs(gradients_sum_update))

        if self._log_scaling:
            log_scaling_update = tf.minimum(log_scaling, tf.square(L_update)/inverse_eta_squared_update)
        else:
            log_scaling_update = log_scaling

        absolute_regret_update = absolute_regret + offset*tf.abs(grad)
        grad_norm_sum_update = grad_norm_sum + tf.abs(grad)

        scalings_update = tf.minimum(scalings, max_grad_norm_update/tf.reduce_sum(L_update))

        offset_update = -tf.sign(gradients_sum_update) * log_scaling_update * scalings_update\
            * (tf.exp(tf.rsqrt(inverse_eta_squared_update) * self._k_inv
            * tf.abs(gradients_sum_update)) - 1.0) + absolute_regret_update/grad_norm_sum_update

        var_update = var + offset_update - offset

        gradients_sum_update_op = state_ops.assign(gradients_sum, gradients_sum_update)
        L_update_op = state_ops.assign(L, L_update)
        inverse_eta_squared_update_op = state_ops.assign(inverse_eta_squared,
                                                         inverse_eta_squared_update)
        log_scaling_update_op = state_ops.assign(log_scaling, log_scaling_update)
        var_update_op = state_ops.assign(var, var_update)

        absolute_regret_update_op = state_ops.assign(absolute_regret, absolute_regret_update)
        grad_norm_sum_update_op = state_ops.assign(grad_norm_sum, grad_norm_sum_update)

        scalings_update_op = state_ops.assign(scalings, scalings_update)

        max_grad_norm_update_op = state_ops.assign(max_grad_norm, max_grad_norm_update)

        with ops.control_dependencies([var_update_op]):
            offset_update_op = state_ops.assign(offset, offset_update)
            absolute_regret_update_op = state_ops.assign(absolute_regret, absolute_regret_update)
            grad_norm_sum_update_op = state_ops.assign(grad_norm_sum, grad_norm_sum_update)

        return control_flow_ops.group(*[gradients_sum_update_op,
                             L_update_op,
                             inverse_eta_squared_update_op,
                             log_scaling_update_op,
                             var_update_op,
                             offset_update_op,
                             absolute_regret_update_op,
                             grad_norm_sum_update_op,
                             scalings_update_op,
                             max_grad_norm_update_op])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

class FreeRexSphere(OptimizerWithAggregates):
    '''FreeExp Learner that uses a full L2 update'''
    def __init__(self, k_inv=1.0/np.sqrt(5), epsilon=1e-8, use_locking=False, name='FreeRexSphere'):
        '''
        constructs a new freerex optimizer
        '''
        super(FreeRexSphere, self).__init__(use_locking, name)
        self._epsilon = epsilon
        self._k_inv = k_inv
        self._inverse_eta_squared = tf.Variable(self._epsilon**2)
        self._L = tf.Variable(epsilon)
        self._log_scaling = tf.Variable(1.0)
        self._log_scaling_update = None
        self._inverse_eta_squared_update = None
        self._grad_sum_norm_squared = None
        self._grad_norm_squared = None

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                gradients_sum = constant_op.constant(0, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)
                L = constant_op.constant(0.0)

                inverse_eta_squared = constant_op.constant(self._epsilon*self._epsilon)

                offset = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)

            self._get_or_make_slot(v, gradients_sum, "gradients_sum", self._name)
            self._get_or_make_slot(v, L, "L", self._name)
            self._get_or_make_slot(v, inverse_eta_squared, "inverse_eta_squared", self._name)
            self._get_or_make_slot(v, offset, "offset", self._name)
            self._get_or_make_slot(v, offset, "last_offset", self._name)

    def _prepare_aggregates(self, grads_and_vars):
        self._grad_norm_squared = constant_op.constant(0.0)
        self._grad_sum_norm_squared = constant_op.constant(0.0)
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            self._grad_norm_squared += 2*tf.nn.l2_loss(grad)
            self._grad_sum_norm_squared += 2 * tf.nn.l2_loss(self.get_slot(var,"gradients_sum") + grad)
        L_update = tf.maximum(self._L, tf.sqrt(self._grad_norm_squared))
        self._inverse_eta_squared_update = tf.maximum(self._inverse_eta_squared \
                                        + 2 * self._grad_norm_squared, L_update * tf.sqrt(self._grad_norm_squared))
        self._log_scaling_update = tf.maximum(self._log_scaling, self._inverse_eta_squared_update/tf.square(L_update))
        log_scaling_update_op = state_ops.assign(self._log_scaling, self._log_scaling_update)
        L_update_op = state_ops.assign(self._L, L_update)
        inverse_eta_squared_update_op = state_ops.assign(self._inverse_eta_squared,
                                                         self._inverse_eta_squared_update)

        update_op = control_flow_ops.group(L_update_op, inverse_eta_squared_update_op, log_scaling_update_op)

        return update_op

    def _apply_dense(self, grad, var):
        gradients_sum = self.get_slot(var, "gradients_sum")
        offset = self.get_slot(var, "offset")
        gradients_sum_update = gradients_sum + grad
        normalized_gradients_sum = \
            gradients_sum_update/(self._log_scaling_update * tf.sqrt(self._grad_sum_norm_squared)+self._epsilon)
        offset_update = -normalized_gradients_sum \
            * (tf.exp(tf.rsqrt(self._inverse_eta_squared_update) * self._k_inv \
                      * tf.sqrt(self._grad_sum_norm_squared)) - 1.0)

        var_update = var + offset_update - offset


        gradients_sum_update_op = state_ops.assign(gradients_sum, gradients_sum_update)
        var_update_op = state_ops.assign(var, var_update)
        with ops.control_dependencies([var_update_op]):
            offset_update_op = state_ops.assign(offset, offset_update)


        return control_flow_ops.group(*[gradients_sum_update_op,
                             offset_update_op,
                             var_update_op])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

class FreeRexLayerWise(Optimizer):
    '''FreeExp Learner that works on each tensorflow variable independently.'''
    def __init__(self, k_inv=1.0/np.sqrt(5), epsilon=1e-6, use_locking=False, name='FreeRexLayerWise'):
        '''
        constructs a new freerex optimizer
        '''
        super(FreeRexLayerWise, self).__init__(use_locking, name)
        self._epsilon = epsilon
        self._k_inv = k_inv

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                gradients_sum = constant_op.constant(0, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)

                L = constant_op.constant(self._epsilon)

                inverse_eta_squared = constant_op.constant(self._epsilon**2)

                log_scaling = constant_op.constant(1.0)

                offset = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)

            self._get_or_make_slot(v, gradients_sum, "gradients_sum", self._name)
            self._get_or_make_slot(v, L, "L", self._name)
            self._get_or_make_slot(v, inverse_eta_squared, "inverse_eta_squared", self._name)
            self._get_or_make_slot(v, log_scaling, "log_scaling", self._name)
            self._get_or_make_slot(v, offset, "offset", self._name)

    def _apply_dense(self, grad, var):
        gradients_sum = self.get_slot(var, "gradients_sum")
        L = self.get_slot(var, "L")
        inverse_eta_squared = self.get_slot(var, "inverse_eta_squared")
        log_scaling = self.get_slot(var, "log_scaling")
        offset = self.get_slot(var, "offset")

        gradients_sum_update = gradients_sum + grad
        L_update = tf.maximum(L, tf.sqrt(2*tf.nn.l2_loss(grad)))

        inverse_eta_squared_update = tf.maximum(inverse_eta_squared + 2*tf.nn.l2_loss(grad), 
                                                L_update * tf.sqrt(2*tf.nn.l2_loss(gradients_sum_update)))

        log_scaling_update = tf.minimum(log_scaling, tf.square(L_update)/inverse_eta_squared_update)

        normalized_gradients_sum = tf.reshape(tf.nn.l2_normalize(tf.reshape(gradients_sum_update, 
                                                                            [-1]), 0),
                                              gradients_sum_update.get_shape())

        offset_update = -normalized_gradients_sum * log_scaling\
            * (tf.exp(tf.rsqrt(inverse_eta_squared_update) * self._k_inv
            * tf.sqrt(2*tf.nn.l2_loss(gradients_sum_update))) - 1.0)

        var_update = var + offset_update - offset

        
        gradients_sum_update_op = state_ops.assign(gradients_sum, gradients_sum_update)
        L_update_op = state_ops.assign(L, L_update)
        inverse_eta_squared_update_op = state_ops.assign(inverse_eta_squared,
                                                         inverse_eta_squared_update)
        log_scaling_update_op = state_ops.assign(log_scaling, log_scaling_update)
        var_update_op = state_ops.assign(var, var_update)
        with ops.control_dependencies([var_update_op]):
            offset_update_op = state_ops.assign(offset, offset_update)

        return control_flow_ops.group(*[gradients_sum_update_op,
                             L_update_op,
                             inverse_eta_squared_update_op,
                             log_scaling_update_op,
                             var_update_op,
                             offset_update_op])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

FreeRex = FreeRexDiag # Default use is FreeRexDiag
