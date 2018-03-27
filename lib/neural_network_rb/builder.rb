module NeuralNetworkRb
  class NeuralNetwork
    class Builder

      attr_reader :layers, :named_layers

      def initialize(random_seed = 1234, &block)
        @layers = []
        @named_layers = {}
        @app = nil
        Numo::NArray.srand(random_seed)
        instance_eval(&block) if block_given?
      end

      def use(clazz, options = {}, &block)
        @layers.unshift [clazz, options, block]
      end

      def to_network
        layer = nil
        @layers.each do |clazz, options, block|
          layer_name = options.delete(:name)
          layer = clazz.new(layer, options, &block)
          @named_layers[layer_name] = layer if layer_name
        end
        @app = layer
      end

    end
    
  end
end
