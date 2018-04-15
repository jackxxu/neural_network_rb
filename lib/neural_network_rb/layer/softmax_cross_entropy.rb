require_relative 'base'

module NeuralNetworkRb
  module Layer
    class SoftmaxCrossEntropy < Base

      private 
        
        def grad(input, target)
          input - target        
        end

        def calc(input)
          softmax(input)
        end
      
        def softmax(x)
          e_x = Numo::NMath.exp(x - x.max(axis: 1, keepdims: true))
          e_x / e_x.sum(axis: 1, keepdims: true)
        end

    end
  end
end
