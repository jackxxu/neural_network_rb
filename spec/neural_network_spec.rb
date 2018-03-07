RSpec.describe TensorflowRb::NeuralNetwork do

  describe '::train', focus: true do
    let(:x) { Numo::DFloat[[0,0], [0,1], [1,0], [1,1]] }
    let(:y) { Numo::DFloat[[0],   [1],   [1],   [0]] }
    let(:network) { TensorflowRb::NeuralNetwork.new(3, 0.1)}

    it 'reduces the error for each step' do
      network.input = x
      network.target = y

      error1, error2 = 0, 0
      network.train() {|n| error1 = TensorflowRb.l2error(n.target, n.output)}
      network.train() {|n| error2 = TensorflowRb.l2error(n.target, n.output)}

      expect(error1).to be > error2
    end
  end

end