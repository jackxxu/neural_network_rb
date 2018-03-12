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

    def accuracy(values, labels)
      size = values.shape[0]
      hits = 0
      size.times.each do |i|
        hits += 1 if labels[i] == values[i, true].max_index
      end
      hits.to_f/size
    end
  end
end