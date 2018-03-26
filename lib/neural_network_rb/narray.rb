module NeuralNetworkRb

  class << self

    def shuffle(data, target, seed)
      sample_size = data.shape[0]
      new_order = Numo::DFloat[*(0..sample_size-1).to_a.shuffle(random: Random.new(seed))]
      [rows(data, new_order), rows(target, new_order)]
    end

    def split(data, ratio)
      sample_size = (data.shape[0] * ratio).to_i
      [rows(data, 0..sample_size-1), rows(data, sample_size..-1)]
    end

    def rows(data, obj)
      data[obj, *Array.new(data.ndim-1, true)]
    end

  end

end