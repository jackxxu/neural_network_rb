require_relative 'activations'

module NeuralNetworkRb
  module Activations
    class Dot

      attr_reader :width, :height
      attr_reader :weight, :bias

      def initialize(next_layer, width: width, learning_rate: learning_rate)
        @width = width
        @learning_rate = learning_rate
        @next_layer = next_layer
      end

      def train(input, target)
        output = calc(input)
        next_grad = @next_layer.train(output, target)

        # learn by adjusting the weight and bias
        @weight -= input.transpose.dot(next_grad) * @learning_rate
        @bias   -= next_grad.sum(axis=0) * @learning_rate
        
        next_grad.dot(@weight.transpose)
      end

      def predict(input)
        @next_layer.predict(calc(input))
      end

      def calc(input)
        if @height.nil?
          @height = input.shape[1]
          @weight = init_weight(@width, @height)
          @bias   = init_bias(@width)
        end
        Numo::Linalg.matmul(input, @weight) + @bias
      end
      
      private 

        def init_weight(width, height)
           balanced_distr = Numo::DFloat.new( height, width).rand - 
                            Numo::DFloat.ones(height, width)/2
           balanced_distr * 0.01 / Math.sqrt(width)
        end

        def init_bias(width)
          balanced_distr = Numo::DFloat.new(width).rand - 
                           Numo::DFloat.ones(width)/2
          balanced_distr * 0.01 / Math.sqrt(width)
       end

    end
  end
end
