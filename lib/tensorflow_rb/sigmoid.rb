module TensorflowRb
  extend self

  def sigmoid(x)
    1/(1 + Math.exp(-x))
  end

  def sigmoid_prime(x)
    s = sigmoid(x)
    s * (1-s)
  end

end