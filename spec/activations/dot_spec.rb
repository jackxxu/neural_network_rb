RSpec.describe NeuralNetworkRb::Activations::Dot do

  let(:width) { 20 }
  let(:layer) { NeuralNetworkRb::Activations::Dot.new(nil, width: width) }

  describe 'initialization', focus: true do

    it 'can initialize' do
      expect(layer).not_to be_nil    
    end

    it 'sets the proper width' do
      expect(layer.width).to eql(width)
    end

    it 'lazy sets weight' do
      expect(layer.weight).to be_nil
    end
  end

  describe 'processing', focus: true do
    let(:input)  { Numo::DFloat[[0,0], [0,1], [1,0], [1,1]] }
    let(:height) { 2 }
    before do
      @result = layer.calc(input)
    end

    it 'initializes the height' do
      expect(layer.height).to eql(height)
    end

    it 'initializes the weight' do
      expect(layer.weight.shape).to eql([height, width])
    end

    it 'has weight of total less than 0.03' do
      expect(layer.weight.to_a.flatten.map(&:abs).inject(:+)).to be < 0.03
    end

    it 'results in 4 * 20 matrix' do
      expect(@result.shape.to_a).to eql([4, 20])
    end
  end

end