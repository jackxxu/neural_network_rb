RSpec.describe TensorflowRb::NeuralNetwork do

  describe 'TensorflowRb.shuffle' do
    let(:data)   { Numo::DFloat.new(10, 2).seq }
    let(:target) { Numo::DFloat.new(10).seq }

    it 'keep the same array length' do
      result = TensorflowRb.shuffle(data, target)
      p result
      expect(data.shape).to eql(result[0].shape)
      expect(target.shape).to eql(result[1].shape)
    end
  end
  
  describe 'TensorflowRb.split' do
    context '2 dimensions' do
      let(:data)   { Numo::DFloat.new(10, 2).seq }

      it 'splits by the ratio' do
        train, test = TensorflowRb.split(data, 0.9)
        expect(train.shape).to eql([9, 2])
        expect(test.shape).to eql([1, 2])
      end        
    end

    context '1 dimension' do
      let(:data)   { Numo::DFloat.new(10).seq }

      it 'splits by the ratio' do
        train, test = TensorflowRb.split(data, 0.9)
        expect(train.shape).to eql([9])
        expect(test.shape).to eql([1])
      end        
    end
  end
end

