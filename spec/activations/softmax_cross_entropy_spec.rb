RSpec.describe NeuralNetworkRb::Layer::SoftmaxCrossEntropy do

  let(:width) { 20 }
  let(:layer) { NeuralNetworkRb::Layer::SoftmaxCrossEntropy.new(nil) }

  describe 'initialization', focus: true do
    it 'can initialize' do
      expect(layer).not_to be_nil    
    end
  end

  # describe 'processing', focus: true do
  #   let(:input)  { Numo::DFloat[[1, 2], [2, 1], [3, 4]] }
  #   let(:target) { Numo::DFloat[[0, 1], [1, 0], [0, 1]] }

  #   before do
  #     @result = layer.calc(input, target)
  #     @grad   = layer.grad(input, target)
  #   end

  #   it 'returns error with the proper dimensions' do
  #     expect(@result).to eql(0.4698925312773342)
  #   end
    
  #   it 'calculates the gradient' do
  #     expect(NArrayComparator.equal(@grad, input-target)).to equal(true)
  #   end
  # end
end