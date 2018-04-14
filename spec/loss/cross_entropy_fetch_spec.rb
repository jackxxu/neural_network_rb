RSpec.describe NeuralNetworkRb::Layer::CrossEntropyFetch, focus: true do

  let(:epoches_count) { 100 }
  let(:input)         { Numo::DFloat[[0, 0], [0, 1], [1, 0], [1, 1]] }
  let(:target)        { Numo::DFloat[[1, 0], [0, 1], [0, 1], [1, 0]] }
  let(:test)          { Numo::DFloat[[0, 0], [0, 1], [1, 0], [1, 1]] }

  context 'multiple layer network' do
    let(:x) { nil }
    let(:rack) {  
      NeuralNetworkRb::NeuralNetwork::Builder.new do 
        use NeuralNetworkRb::Layer::Dot, width: 20, name: :dot1, learning_rate: 1
        use NeuralNetworkRb::Layer::Sigmoid
        use NeuralNetworkRb::Layer::Dot, width: 2, name: :dot2, learning_rate: 1
        use NeuralNetworkRb::Layer::SoftmaxCrossEntropy
        use NeuralNetworkRb::Layer::CrossEntropyFetch, every: 100 do |error| 
          x = error
        end
      end
    }
    let!(:network) { rack.to_network }

    let(:layer_classes) { rack.layers.map(&:first)  }

    it 'has 4 layers' do
      expect(rack.layers.length).to eql(5)
    end

    it 'has all the layers in the specified order' do
      expect(layer_classes).to eq([
        NeuralNetworkRb::Layer::CrossEntropyFetch,
        NeuralNetworkRb::Layer::SoftmaxCrossEntropy,
        NeuralNetworkRb::Layer::Dot,
        NeuralNetworkRb::Layer::Sigmoid,
        NeuralNetworkRb::Layer::Dot
      ])
    end

    it 'has only one named layer' do
      expect(rack.named_layers.length).to eql(2)        
    end

    describe '::to_network' do
      let!(:network) { rack.to_network }

      it 'has only one named layer with the specified name' do
        expect(rack.named_layers.keys).to eql([:dot2, :dot1])        
      end

      describe '::train' do
        before do 
          @grad = network.train(input, target)
        end

        it 'has a valid gradient' do
          expect(@grad.shape).to eql([4, 2])
        end

        it 'changes the weight' do
          before_weight = rack.named_layers[:dot1].weight
          network.train(input, target)
          after_weight = rack.named_layers[:dot1].weight
          expect(NArrayComparator.equal(before_weight, after_weight)).to be(false)
        end

        it 'reduces the error gradually' do
        end
        
      end
    end
  end
end

