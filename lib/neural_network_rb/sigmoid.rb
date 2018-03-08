module NeuralNetworkRb

  E = Math.exp(1)

  class << self 
    def sigmoid(x)
      1/(1 + E**(-x))
    end

    def sigmoid_prime(x)
      s = sigmoid(x)
      s * (1-s)
    end
  end

end