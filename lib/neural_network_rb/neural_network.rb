module NeuralNetworkRb

  class NeuralNetwork

    attr_reader :target, :output, :epoch
    
    def input=(input)
      @input = input/255.0
      input_width = input.shape[1]
      @w1 = (Numo::DFloat.new(input_width,  @neurons_count).rand - 
             Numo::DFloat.ones(input_width,  @neurons_count)/2) * 0.01 / Math.sqrt(input_width)
      @b1 = Numo::DFloat.zeros(@neurons_count)
    end

    def target=(target)
      @target = target
      output_width = target.shape[1]
      @w2 = (Numo::DFloat.new(@neurons_count, output_width).rand -
             Numo::DFloat.ones(@neurons_count, output_width)/2) * 0.01 / Math.sqrt(@neurons_count)
      @b2 = Numo::DFloat.zeros(output_width)
    end

    def initialize(neurons_count, learning_rate, random_seed)
      @neurons_count = neurons_count
      @learning_rate = learning_rate
      @epoch = 0
      Numo::NArray.srand(random_seed)
    end

    def fit()
      forward
      backprop
      @epoch += 1 
      yield self if block_given?
    end

    def forward
      # forward
      @hidden = NeuralNetworkRb.sigmoid(
                  Numo::Linalg.matmul(@input, @w1) + 
                  @b1)
      @output = NeuralNetworkRb.softmax(
                  Numo::Linalg.matmul(@hidden, @w2) + 
                  @b2)
    end

    def backprop
      # backward 
      dZ   = NeuralNetworkRb.plain_diff(@target,  @output) 
      @w2 += Numo::Linalg.matmul(@hidden.transpose, dZ) * @learning_rate
      dH   = Numo::Linalg.matmul(dZ, @w2.transpose) * NeuralNetworkRb.sigmoid_prime(@hidden)
      @w1 += Numo::Linalg.matmul(@input.transpose, dH) * @learning_rate
    end

    def predict(input)
      hidden = NeuralNetworkRb.sigmoid(Numo::Linalg.matmul(input/255.0, @w1))
      output = NeuralNetworkRb.softmax(Numo::Linalg.matmul(hidden, @w2))
      output
    end
  end
end