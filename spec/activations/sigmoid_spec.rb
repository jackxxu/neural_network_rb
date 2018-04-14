RSpec.describe NeuralNetworkRb::Layer::Sigmoid do

  let(:next_layer) { double("foo", train: 1, grad: 1 ) }
  let(:rack) { NeuralNetworkRb::Layer::Sigmoid.new(next_layer) }

  describe '.calc', focus: true do

    let(:result) { rack.calc(x) }

    context 'negative infinity' do
      let(:x) { Numo::DFloat[[-10000, 0], [-10000, 0]] }
      it 'outputs close to 0' do
        expect(result.to_a).to eql([[0.0, 0.5], [0.0, 0.5]])
      end
    end

    context 'positive infinity' do
      let(:x) { Numo::DFloat[[10000, 0], [10000, 0]] }
      it 'outputs close to 1' do
        expect(result.to_a).to eql([[1.0, 0.5], [1.0, 0.5]])
      end
    end

    context 'zero' do
      let(:x) { Numo::DFloat[[0], [0]] }
      it 'outputs 0.5'  do
        expect(result.to_a).to eql([[0.5], [0.5]]) 
      end
    end
  end

  describe '.grad', focus: true do

    let(:result) { rack.grad( rack.calc(x) ) }

    context 'negative infinity' do
      let(:x) { Numo::DFloat[[-10000, 0], [-10000, 0]] }
      it 'outputs close to 0' do
        expect(result.to_a).to eq([[0, 0.25], [0, 0.25]])
      end
    end

    context 'positive infinity' do
      let(:x) { Numo::DFloat[[10000, 0], [10000, 0]] }
      it 'outputs close to 0' do
        expect(result.to_a).to eq([[0, 0.25], [0, 0.25]])
      end
    end

    context 'zero' do
      let(:x) { Numo::DFloat[[0], [0]] }
      it 'calculate sigmoid output' do
        expect(result.to_a).to eq([[0.25], [0.25]])
      end
    end

    context 'vector' do
      let(:x) { Numo::DFloat[[0, 1]] }
      it 'calculate sigmoid output' do
        expect(result.to_a).to eq([[0.25, 0.19661193324148185]])
      end      
    end

  end

end
