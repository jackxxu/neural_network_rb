module NeuralNetworkRb

  def self.softmax(array)
    exp_array = array.map {|x| Math.exp(x)}
    exp_array = exp_array/exp_array.sum
  end

end