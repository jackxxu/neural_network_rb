require_relative 'base'

module NeuralNetworkRb
  module Layer
    class Sigmoid < Base

      def calc(x)
        1 / (1+ Numo::NMath.exp(-x))
      end
  
      def grad(output, _)
        output * (1 - output)
      end
    end
  end
end
