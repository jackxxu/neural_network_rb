module TensorflowRb

  class << self

    def shuffle(data, target)
      sample_size = data.shape[0]
      new_order = Numo::DFloat[*(0..sample_size-1).to_a.shuffle]
      [data[*subset(data, new_order)], target[*subset(target, new_order)]]
    end

    def split(data, ratio)
      sample_size = (data.shape[0] * ratio).to_i
      [data[*subset(data, 0..sample_size-1)], data[*subset(data, sample_size..-1)]]
    end
    
    private

      def subset(data, obj)
        Array.new(data.ndim, true).tap do |x|
          x[0] = obj
        end
      end
  end

end