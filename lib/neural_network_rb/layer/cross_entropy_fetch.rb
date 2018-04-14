module NeuralNetworkRb
  module Layer
    class CrossEntropyFetch

      attr_reader :epoch

      def initialize(next_layer, options = {}, &block)
        @next_layer = next_layer
        @every = options[:every]
        @block = block if block_given?
        @epoch = 0
      end

      def train(input, target)
        error = cross_entropy(input, target)
        @block.call(@epoch, error) if @block && (@epoch % @every == 0)
        @epoch +=1
        @next_layer.nil? ? 1 : @next_layer.train(input, target)
      end

      def predict(input)
        @next_layer.nil? ? input : @next_layer.predict(input)
      end

      private

        def cross_entropy(input, target)
          - (target * input.map {|x| Math.log(x)}).sum/input.ndim
        end

    end
  end
end
