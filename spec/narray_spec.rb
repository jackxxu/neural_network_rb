RSpec.describe NeuralNetworkRb::NeuralNetwork do

  describe 'NeuralNetworkRb.shuffle' do
    let(:data)   { Numo::DFloat.new(10, 2).seq }
    let(:target) { Numo::DFloat.new(10).seq }

    it 'keep the same array length' do
      result = NeuralNetworkRb.shuffle(data, target, 1234)
      expect(data.shape).to eql(result[0].shape)
      expect(target.shape).to eql(result[1].shape)
    end
  end
  
  describe 'NeuralNetworkRb.split' do
    context '2 dimensions' do
      let(:data)   { Numo::DFloat.new(10, 2).seq }

      it 'splits by the ratio' do
        train, test = NeuralNetworkRb.split(data, 0.9)
        expect(train.shape).to eql([9, 2])
        expect(test.shape).to eql([1, 2])
      end        
    end

    context '1 dimension' do
      let(:data)   { Numo::DFloat.new(10).seq }

      it 'splits by the ratio' do
        train, test = NeuralNetworkRb.split(data, 0.9)
        expect(train.shape).to eql([9])
        expect(test.shape).to eql([1])
      end        
    end
  end
end

