RSpec.describe NeuralNetworkRb::MNIST, focus: true do

  context 'downloads training images and labels' do
    before do
      @training_set = NeuralNetworkRb::MNIST.training_set        
    end

    it 'download file from the mnist site' do
      NeuralNetworkRb::MNIST::TRAIN_FILE_NAMES.each do |file|
        expect(File.exists?(file)).to be true
      end
    end
      
    it 'returns an instance of MNIST' do
      expect(@training_set).to be_instance_of(NeuralNetworkRb::MNIST)
    end

    it 'has the 60000 training labels' do 
      expect(@training_set.labels.shape).to eql([60000])
    end

    it 'has the 60000 training images of 784 bits each' do 
      expect(@training_set.data.shape).to eql([60000, 784])
    end
  end

  context 'downloads test images and labels' do
    before do
      @test_set = NeuralNetworkRb::MNIST.test_set        
    end

    it 'download file from the mnist site' do
      NeuralNetworkRb::MNIST::TEST_FILE_NAMES.each do |file|
        expect(File.exists?(file)).to be true
      end
    end
      
    it 'returns an instance of MNIST' do
      expect(@test_set).to be_instance_of(NeuralNetworkRb::MNIST)
    end

    it 'has the 10000 test labels' do 
      expect(@test_set.labels.shape).to eql([10000])
    end

    it 'has the 10000 test images of 784 bits each' do 
      expect(@test_set.data.shape).to eql([10000, 784])
    end
  end

end

