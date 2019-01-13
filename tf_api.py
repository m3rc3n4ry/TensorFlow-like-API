import numpy as np

class Graph():
    def __init__(self):
        """
        creates four lists of objects that are present in a computation graph
        """
        self.operations = []
        self.placeholders = []
        self.constants = []
        self.variables = []

    def as_default(self):
        """
        called to create a global variable for storing current graph instance
        """
        global _default_graph
        _default_graph = self


class Operation():
    """
    An operation is characterized as follows:

    - it has a list of input_nodes
    - implements a forward function
    - implements a backward function
    - remembers its output
    - adds itself to the default graph

    The input nodes is a list of Tensors (≥ 0) that are going into this operation.

    Both forward and backward are only placeholder methods and they must be
    implemented by every specific operation. In our implementation, forward
    is called during the forward pass (or forward-propagation) which computes
    the output of the operation, whereas backward is called during the
    backward pass (or backpropagation) where we calculate the gradient of
    the operation with respect to each input variable.
    """
    def __init__(self, input_nodes=None):
        self.input_nodes = input_nodes
        self.output = None

        # append operation to the list of operations in default graph
        _default_graph.operations.append(self)

    def forward(self):
        pass

    def backward(self):
        pass


class BinaryOperation(Operation):
    """
    To make our life a little bit easier and to avoid unnecessary code
    duplication, BinaryOperation just takes care of initializing a and b as
    input nodes.
    """
    def __init__(self, a, b):
        super().__init__([a, b])


class add(BinaryOperation):
    """
    Computes a + b, element-wise
    """
    def forward(self, a, b):
        return a + b

    def backward(self, upstream_grad):
        raise NotImplementedError

class multiply(BinaryOperation):
    """
    Computes a * b, element-wise
    """
    def forward(self, a, b):
        return a * b

    def backward(self, upstream_grad):
        raise NotImplementedError

class divide(BinaryOperation):
    """
    Returns the true division of the inputs, element-wise
    """
    def forward(self, a, b):
        return np.true_divide(a, b)

    def backward(self, upstream_grad):
        raise NotImplementedError

class matmul(BinaryOperation):
    """
    Multiplies matrix a by matrix b, producing a * b
    """
    def forward(self, a, b):
        return a.dot(b)

    def backward(self, upstream_grad):
        raise NotImplementedError


class Placeholder():
    """
    It’s not being initialized with a value, hence the name, and only appends itself to the default graph.
    The value for the placeholder is provided using the feed_dict optional argument to Session.run()
    """
    def __init__(self):
        self.value = None
        _default_graph.placeholders.append(self)


class Constant():
    """
    Constants cannot be changed once initialized.
    """
    def __init__(self, value=None):
            self.__value = value
            _default_graph.constants.append(self)

    @property
    def value(self):
        """
        we are defining a property __value inside the class constructor.
        Because of the double underscore in the property name, Python will
        rename the property internally to something like _Constant__value,
        so it prefixes the property with the class name. This feature is
        actually meant to prevent naming collisions when working with inheritance.
        """
        return self.__value

    @value.setter
    def value(self, value):
        raise ValueError("Cannot reassign value.")

class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        _default_graph.variables.append(self)


def topology_sort(operation):
    ordering = []
    visited_nodes = set()

    def recursive_helper(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                if input_node not in visited_nodes:
                    recursive_helper(input_node)

        visited_nodes.add(node)
        ordering.append(node)
    
    # start recursive depth-first search
    recursive_helper(operation)

    return ordering


class Session():
    def run(self, operation, feed_dict={}):
        nodes_sorted = topology_sort(operation)
        
        for node in nodes_sorted:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable or type(node) == Constant:
                node.output = node.value
            else:
                inputs = [node.output for node in node.input_nodes]
                node.output = node.forward(*inputs)

        return operation.output 

