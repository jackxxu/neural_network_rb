RSpec.describe NeuralNetworkRb do

  describe 'NeuralNetworkRb.sigmoid' do

    let(:result) { NeuralNetworkRb.sigmoid(x) }

    context 'negative infinity' do
      let(:x) { Numo::DFloat[[-10000, 0], [-10000, 0]] }
      it 'outputs close to 0' do
        expect(result.to_a).to eql([[0.0, 0.5], [0.0, 0.5]])
      end
    end

    context 'positive infinity' do
      let(:x) { Numo::DFloat[[10000, 0], [10000, 0]] }
      it 'outputs close to 1' do
        p result
        expect(result.to_a).to eql([[0.5, 0.0], [0.5, 0.0]])
      end
    end

    context 'zero' do
      let(:x) { Numo::DFloat[[0], [0]] }
      it 'outputs 0.5'  do
        expect(result.to_a).to eql([[0.5], [0.5]]) 
      end
    end
  end

  describe 'NeuralNetworkRb.sigmoid_prime' do

    let(:result) { NeuralNetworkRb.sigmoid_prime(x) }

    context 'negative infinity' do
      let(:x) { Numo::DFloat[[-10000, 0], [-10000, 0]] }
      it 'outputs close to 0' do
        expect(result.to_a).to eq([[0, 0.25], [0, 0.25]])
      end
    end

    context 'positive infinity' do
      let(:x) { Numo::DFloat[[10000, 0], [10000, 0]] }
      it 'outputs close to 0' do
        expect(result.to_a).to eq([[0.25, 0], [0.25, 0]])
      end
    end

    context 'zero' do
      let(:x) { Numo::DFloat[[0], [0]] }
      it 'calculate sigmoid output' do
        expect(result.to_a).to eq([[0.25], [0.25]])
      end
    end
  end

end
