RSpec.describe NeuralNetworkRb::Embeddings do

  describe '#one_hot' do
    let(:y)           { Numo::Int8[1, 2, 3, 4] }
    let(:class_count) { 4 }
    let(:expected)    { Numo::Int8[[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]] }
    it 'produces one-hot matrix', focus: true do
      result = NeuralNetworkRb::Embeddings.one_hot(y, class_count)
      expect(result.to_a).to eql(expected.to_a)
    end
  end

  
end


