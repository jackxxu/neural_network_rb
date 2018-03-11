module NeuralNetworkRb

  def self.softmax(array)
    max_array = array.max(1)
    exp_array = array.map_with_index {|a, i| a - max_array[i]}.map {|x| Math.exp(x)}
    sum_array = exp_array.sum(1)
    exp_array.map_with_index {|a, i| a/sum_array[i]}
  end

end