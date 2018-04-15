module NeuralNetworkRb
  module Layer
    class CrossEntropyFetch < Base

      attr_reader :epoch

      def initialize(next_layer, options = {}, &block)
        @next_layer = next_layer
        @options = options
        @every = @options[:every]
        @block = block if block_given?
        @epoch = 0
      end

      def train(input, target)
        error = calc(input, target)
        @block.call(@epoch, error) if @block && (@epoch % @every == 0)
        @epoch +=1
        @next_layer.nil? ? 1 : @next_layer.train(input, target)
      end

      private

        def calc(input, target)
          - (target * input.map {|x| Math.log(x)}).sum/input.ndim
        end

    end
  end
end
