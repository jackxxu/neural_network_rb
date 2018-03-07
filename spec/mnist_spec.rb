RSpec.describe TensorflowRb::MNIST do

  describe '::download' do
    before do
      @mist = TensorflowRb::MNIST.download!
    end

    it 'download file from the mnist site' do
      TensorflowRb::MNIST::FILE_NAMES.each do |file|
        expect(File.exists?(file)).to be true
      end
    end

    it 'returns an instance of MNIST' do
      expect(@mist).to be_instance_of(TensorflowRb::MNIST)
    end
  end


end

