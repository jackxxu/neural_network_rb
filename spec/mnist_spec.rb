RSpec.describe TensorflowRb::MNIST do

  before do
    @mist = TensorflowRb::MNIST.download!
  end

  describe '.download' do

    it 'download file from the mnist site' do
      TensorflowRb::MNIST::FILE_NAMES.each do |file|
        expect(File.exists?(file)).to be true
      end
    end

    it 'returns an instance of MNIST' do
      expect(@mist).to be_instance_of(TensorflowRb::MNIST)
    end
  end

  describe '#training_labels' do
    let(:training_labels) { @mist.training_labels }

    it 'has the 60000 training labels' do 
      expect(training_labels.shape).to eql([60000])
    end
  end


  describe '#training_images', focus: true do
    let(:training_images) { @mist.training_images }

    it 'has the 60000 training images of 784 bits each' do 
      expect(training_images.shape).to eql([60000, 784])
    end
  end

  describe '#test_labels' do
    let(:test_labels) { @mist.test_labels }

    it 'has the 10000 test labels' do 
      expect(test_labels.shape).to eql([10000])
    end
  end


  describe '#test_images', focus: true do
    let(:test_images) { @mist.test_images }

    it 'has the 10000 test images of 784 bits each' do 
      expect(test_images.shape).to eql([10000, 784])
    end
  end


end

