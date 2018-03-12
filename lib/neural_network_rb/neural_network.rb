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

    def initialize(neurons_count, learning_rate, random_seed)
      @neurons_count = neurons_count
      @learning_rate = learning_rate
      @epoch = 0
      Numo::NArray.srand(random_seed)
    end

    def fit()
      # forward
      @hidden = NeuralNetworkRb.sigmoid(Numo::Linalg.matmul(@input, @w_hidden))

      @output = Numo::Linalg.matmul(@hidden, @w_output)
      o = NeuralNetworkRb.softmax(@output)

      # calculate error
      error_algorithm = :plain_diff
      error = NeuralNetworkRb.send(error_algorithm, @target,  @output)

      # backward 
      dZ = error * @learning_rate

      @w_output += Numo::Linalg.matmul(@hidden.transpose, dZ)
      dH         = Numo::Linalg.matmul(dZ, @w_output.transpose) * NeuralNetworkRb.sigmoid_prime(@hidden)
      @w_hidden += Numo::Linalg.matmul(@input.transpose, dH)

      @epoch += 1 
      yield self if block_given?
    end

  end
end