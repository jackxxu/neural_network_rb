module NeuralNetworkRb
  module Layer
    class Base
      def initialize(next_layer, options = {})
        @next_layer = next_layer
        @options = options
      end

      def train(input, target)
        output = calc(input)
        @next_layer.nil? ? 1 : grad(output, target) * @next_layer.train(output, target)
      end

      def predict(input)
        @next_layer.nil? ? input : @next_layer.predict(calc(input))          
      end
    end
  end
end
