RSpec.describe NeuralNetworkRb::MNIST do

  describe 'training data' do
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

    describe '#batchize' do
      let(:batches_count) { 300 }
      before do 
        @batches = @training_set.batches(batches_count)
      end

      it 'create a batches array of proper size' do
        expect(@batches.length).to be(batches_count)
      end

      it 'has each array element  data and labels' do
        expect(@batches[0].length).to be(2)        
      end

      it 'has data element of proper size' do
        expect(@batches[0][0].shape).to eql([200, 784])        
        expect(@batches[299][0].shape).to eql([200, 784])        
      end    

      it 'has labels element of proper size' do
        expect(@batches[0][1].shape).to eql([200])        
        expect(@batches[299][1].shape).to eql([200])        
      end    

    end

    describe '#one_hot embedding labels' do
      before do 
        @training_set2 = @training_set.clone
        @training_set2.embed_labels!(:one_hot, 10)
      end

      it 'changes the labels shape' do
        expect(@training_set2.labels.shape).to eq([60000, 10])        
      end

      it 'each one label has an 1 value' do
        expect(@training_set2.labels[0, true].sum).to eq(1)                
      end
    end

    describe '#shuffles' do
      before do
        @pre_shuffle_data = @training_set.data[0..5, true].copy
        @pre_shuffle_labels = @training_set.labels[0..5].copy
        @training_set.shuffle!
      end

      it 'changes the order of the data' do
        expect(@training_set.data[0..5, true]).not_to eql(@pre_shuffle_data)
      end

      it 'changes the order of the labels' do
        expect(@training_set.labels[0..5]).not_to eql(@pre_shuffle_labels)
      end
    end

    describe '#partition' do
      before do
        @training_set.partition!(0.9)
      end

      it 'changes the order of the data' do
        expect(@training_set.data.shape).to eq([54000, 784])
      end

      it 'changes the shape of the labels' do
        expect(@training_set.labels.shape).to eq([54000])
      end

      it 'creates validation data' do
        expect(@training_set.validation_data.shape).to eq([6000, 784])        
      end

      it 'creates validation labels' do
        expect(@training_set.validation_labels.shape).to eq([6000])        
      end
    end

  end

  describe 'test data' do
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

  describe 'train', focus: true do
    let(:embedding) { :one_hot }
    let(:label_classes) { 10 }
    let(:epochs) { 300 }
    let(:neuron_count)  { 20 }
    let(:learning_rate) { 0.05 }
    before do
      @training_set = NeuralNetworkRb::MNIST.training_set
                                            .embed_labels!(embedding, label_classes)
                                            .shuffle!
                                            .partition!(0.9)
      @network = NeuralNetworkRb::NeuralNetwork.new(neuron_count, learning_rate)
    end

    it 'does something' do
      # @training_set.batches(epochs).each do |batch|
      #   data, labels = *batch
      #   puts data.shape.inspect
      # end
    end
  end

end

