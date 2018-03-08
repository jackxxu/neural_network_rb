module NeuralNetworkRb
  module Embeddings
    class << self
      def one_hot(y, class_count)
        Numo::Int8.zeros(y.shape[0], class_count).tap do |matrix|
          y.each_with_index do |v, i|
            matrix[i, v-1] = 1
          end          
        end
      end
    end
  end
end