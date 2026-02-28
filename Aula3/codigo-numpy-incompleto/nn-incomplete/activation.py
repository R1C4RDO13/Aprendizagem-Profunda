#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from layers import Layer

class ActivationLayer(Layer):

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0
    
## O calculo das derivadas, permite-nos calcular qual a contribuição de cada neurónio na função do erro
## Se o valor da derivada estiver a aumentar, significa que o nosso modelo não está a convergir
## E com isto tomar decisões sobre os próximos passos a seguir
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        ## COMPLETE

        s = self.activation_function(input)
        return s * (1 - s)


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):

        ## COMPLETE
        return (input > 0).astype(float)
    
