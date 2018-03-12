module NeuralNetworkRb

  E = Math.exp(1)

  class << self 
    def sigmoid(array)
      max_array = array.max(1)
      array.map_with_index {|a, i| a - max_array[i]}
           .map_with_index {|x, i| 1/(1 + E**(-x))}
    end

    def sigmoid_prime(x)
      s = sigmoid(x)
      s * (1-s)
    end

    def softmax(array)
      max_array = array.max(1)
      exp_array = array.map_with_index {|a, i| a - max_array[i]}.map {|x| Math.exp(x)}
      sum_array = exp_array.sum(1)
      exp_array.map_with_index {|a, i| a/sum_array[i]}
    end
  
  end

end