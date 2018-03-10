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
      t1 = Time.now
      puts @input.shape.inspect
      puts @w_hidden.shape.inspect
      t = @input.dot(@w_hidden)
      puts Time.now - t1
      @hidden = NeuralNetworkRb.sigmoid(t)
      @output = @hidden.dot(@w_output)

      o = @output.shape[0].times.map {|i| NeuralNetworkRb.softmax(output[i, true])}
      @output = Numo::NArray[*o]
      # calculate error
      error_algorithm = :plain_diff
      error = NeuralNetworkRb.send(error_algorithm, @target,  @output)
      # error = @target - @output

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