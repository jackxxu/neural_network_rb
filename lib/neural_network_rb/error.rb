module NeuralNetworkRb
  class << self
    def l2error(v1, v2)
      ((v1 - v2) ** 2).sum
    end

    def l1error(v1, v2)
      (v1 - v2).abs.sum
    end

    def plain_diff(v1, v2)
      v1 - v2
    end

    def cross_entropy(values, labels)
      - (labels * values.map {|x| Math.log(x)}).sum/values.ndim
    end
  end
end