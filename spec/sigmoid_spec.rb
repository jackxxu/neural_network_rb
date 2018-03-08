RSpec.describe NeuralNetworkRb do

  describe 'NeuralNetworkRb.sigmoid' do

    let(:result) { NeuralNetworkRb.sigmoid(x) }

    context 'negative infinity' do
      let(:x) { Numo::DFloat[-10000] }
      it 'outputs close to 0' do
        expect(result[0]).to be < 0.000001
      end
    end


    context 'positive infinity' do
      let(:x) { Numo::DFloat[10000] }
      it 'outputs close to 1' do
        expect(result[0]).to be > 0.99999999
      end
    end

    context 'zero' do
      let(:x) { Numo::DFloat[0] }
      it 'outputs 0.5' do
        expect(result[0]).to be(0.5) 
      end
    end
  end

  describe 'NeuralNetworkRb.sigmoid_prime' do

    let(:result) { NeuralNetworkRb.sigmoid_prime(x) }

    context 'negative infinity' do
      let(:x) { Numo::DFloat[-10000] }
      it 'outputs close to 0' do
        expect(result[0]).to be < 0.000001
      end
    end

    context 'positive infinity' do
      let(:x) { Numo::DFloat[10000] }
      it 'outputs close to 0' do
        expect(result[0]).to be < 0.000001
      end
    end

    context 'zero' do
      let(:x) { Numo::DFloat[0] }
      it 'calculate sigmoid output' do
        expect(result[0]).to be(0.25) 
      end
    end
  end

end
