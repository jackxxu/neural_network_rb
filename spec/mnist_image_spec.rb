RSpec.describe NeuralNetworkRb::MNIST do

  context 'saves the image file as PNG' do
    let(:file_name) { 'test.png' }
    before do
      @mist = NeuralNetworkRb::MNIST.training_set
      pixels = @mist.data[0, true]
      NeuralNetworkRb::MNIST::Image.new(pixels).save_to_file(file_name)
    end
    
    it 'saves to a png file' do
      expect(File.exists?(file_name)).to be true
    end      
  end
end

