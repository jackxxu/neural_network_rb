module NeuralNetworkRb
  module Loss
    class SoftmaxCrossEntropy

      def initialize(next_layer, options = {})
        @next_layer = next_layer
      end

      def train(input, target)
        output = softmax(input)
        grad(output, target) * @next_layer.train(output, target)
      end

      def predict(input)
        @next_layer.predict(softmax(input))  
      end

      def call(input, target)
        error = cross_entropy(softmax(input), target)
        grad(input, target) if target
      end

      private 
        
        def grad(input, target)
          input - target        
        end
      
        def softmax(x)
          e_x = Numo::NMath.exp(x - x.max(axis: 1, keepdims: true))
          e_x / e_x.sum(axis: 1, keepdims: true)
        end

    end
  end
end
