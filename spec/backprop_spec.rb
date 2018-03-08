RSpec.describe NeuralNetworkRb do

  describe 'backprop' do
    let(:x) { Numo::DFloat[[0,0], [0,1], [1,0], [1,1]] }
    let(:y) { Numo::DFloat[[0],   [1],   [1],   [0]] }

    let(:input_size)  { 2 }
    let(:hidden_size) { 3 }
    let(:output_size) { 1 }
    let(:lr)          { 0.1 }

    it 'back propagates' do
      w_hidden = Numo::DFloat.new(input_size,  hidden_size).rand
      w_output = Numo::DFloat.new(hidden_size, output_size).rand
      # forward
      hidden = NeuralNetworkRb.sigmoid(x.dot(w_hidden))
      output = hidden.dot(w_output)
      # calculate error
      error = y - output
      # backward 
      dZ = error * lr

      w_output += hidden.transpose.dot(dZ)
      dH = dZ.dot(w_output.transpose) * NeuralNetworkRb.sigmoid_prime(hidden)
      w_hidden += x.transpose.dot(dH)

      p w_hidden
    end
  end
end