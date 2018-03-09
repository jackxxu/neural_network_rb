module NeuralNetworkRb
  module Embeddings
    class << self
      def one_hot(labels, class_count)
        Numo::Int8.zeros(labels.shape[0], class_count).tap do |matrix|
          labels.each_with_index do |v, i|
            matrix[i, v] = 1
          end          
        end
      end
    end
  end
end