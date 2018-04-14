module NeuralNetworkRb
  module Layer
    class ReLU

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

      def calc(x)
        x * (x > 0)
      end
  
      def grad(output)
        1.0 * (output > 0)
      end
    end
  end
end
