'''
base class for forming optimizers that use information aggregated from all gradients.
'''

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training.optimizer import Optimizer

#my install of tf doesn't have this function, so it's just copied here.
def _get_variable_for(v):
    """Returns the ResourceVariable responsible for v, or v if not necessary."""
    if v.op.type == "VarHandleOp":
        for var in ops.get_collection(ops.GraphKeys.RESOURCES):
            if (isinstance(var, resource_variable_ops.ResourceVariable)
                    and var.handle.op is v.op):
                return var
        raise ValueError("Got %s but  could not locate source variable." % (str(v)))
    return v


class OptimizerWithAggregates(Optimizer):
    '''base class for optimizers that need aggregate data from all gradients'''

    def _prepare_aggregates(self, grads_and_vars):
        '''returns an op that updates aggregate variables'''
        pass

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):

        # Do error checking and create slots. Error checking is copied from base apply_gradients
        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    g = ops.convert_to_tensor_or_indexed_slices(g)
                except TypeError:
                    raise TypeError(
                            "Gradient must be convertible to a Tensor"
                            " or IndexedSlices, or None: %s" % g)
                if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError(
                            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
            converted_grads_and_vars.append((g, v))

        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list = [v for g, v in converted_grads_and_vars if g is not None]
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." %
                                             ([str(v) for _, _, v in converted_grads_and_vars],))
        with ops.control_dependencies(None):
            self._create_slots([_get_variable_for(v) for v in var_list])

        ##### end copypasta code #####

        with ops.name_scope(name, self._name) as name:
            aggregate_op = self._prepare_aggregates(converted_grads_and_vars)

        non_aggregate_updates = super(OptimizerWithAggregates, self).apply_gradients(grads_and_vars, global_step=global_step, name=name)

        apply_updates = control_flow_ops.group(aggregate_op, non_aggregate_updates)

        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        if aggregate_op not in train_op:
            train_op.append(aggregate_op)

        return apply_updates