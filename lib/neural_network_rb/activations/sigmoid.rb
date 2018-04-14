require_relative 'activations'

module NeuralNetworkRb
  module Activations
    class Sigmoid

      def initialize(next_layer, options = {})
        @next_layer = next_layer
      end

      def train(input, target)
        output = calc(input)
        grad(output) * @next_layer.train(output, target)
      end

      def predict(input)
        @next_layer.predict(calc(input))          
      end

      def call(input, target)
        output = calc(input)
        if target
          grad(output) * @next_layer.call(output, target)
        else
          @next_layer.call(output, target)          
        end
      end

      def calc(x)
        1 / (1+ Numo::NMath.exp(-x))
      end
  
      def grad(output)
        # s = calc(x)
        output * (1 - output)
      end
    end
  end
end
