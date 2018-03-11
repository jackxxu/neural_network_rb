RSpec.describe NeuralNetworkRb do

  describe 'NeuralNetworkRb.softmax', focus: true do    
    let(:data) { Numo::DFloat[[2, 4, 6, 8]] }
    let(:result) { NeuralNetworkRb.softmax(data) }

    it 'calculate the neural network output' do
      expect(result.to_a).to eq([[ 0.002144008783584634, 0.01584220117850692, 0.11705891323853293, 0.8649548767993754]])
    end
  end

end
