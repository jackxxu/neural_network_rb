module NeuralNetworkRb

  class NeuralNetwork

    attr_reader :target, :output, :epoch
    
    def input=(input)
      @input = input
      input_width = input.shape[1]
      @w_hidden = Numo::DFloat.new(input_width,  @neurons_count).rand
    end

    def target=(target)
      @target = target
      output_width = target.shape[1]
      @w_output = Numo::DFloat.new(@neurons_count, output_width).rand
    end

    def initialize(neurons_count, learning_rate)
      @neurons_count = neurons_count
      @learning_rate = learning_rate
      @epoch = 0
    end

    def train()
      # forward
      @hidden = NeuralNetworkRb.sigmoid(@input.dot(@w_hidden))
      @output = @hidden.dot(@w_output)

      # calculate error
      error = @target - @output

      # backward 
      dZ = error * @learning_rate

      @w_output += @hidden.transpose.dot(dZ)
      dH = dZ.dot(@w_output.transpose) * NeuralNetworkRb.sigmoid_prime(@hidden)
      @w_hidden += @input.transpose.dot(dH)

      @epoch += 1 
      yield self if block_given?
    end

  end
end