require_relative 'base'

module NeuralNetworkRb
  module Layer
    class ReLU < Base

      def calc(x)
        x * (x > 0)
      end
  
      def grad(output, _)
        1.0 * (output > 0)
      end
    end
  end
end
