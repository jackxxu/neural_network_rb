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
      let(:labels) { Numo::Int16[0, 1, 7, 3] }
      before do 
        @embeddings = NeuralNetworkRb::MNIST.embed_labels(labels, :one_hot, 10)
      end

      it 'add one dimension of 10 columns' do
        expect(@embeddings.shape).to eq([4, 10])        
      end

      it 'embeds proper value' do
        expect(@embeddings.to_a).to eq([
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
        ])                
      end
    end

    describe '#shuffles' do
      before do
        @pre_shuffle_data = @training_set.data[0..5, true].copy
        @pre_shuffle_labels = @training_set.labels[0..5].copy
        @training_set.shuffle!(1)
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

  describe 'train with rack' do
    let(:embedding)     { :one_hot }
    let(:epochs)        { 50000 }
    let(:random_seed)   { 4567 }
    let(:batch_size)    { 40 }
    let(:data_ratio)    { 0.9 }

    let!(:network) { rack.to_network }
    
    before do
      @training_set = NeuralNetworkRb::MNIST.training_set
                                            .normalize!
                                            .shuffle!(random_seed)
                                            .partition!(data_ratio)
      @test_set     = NeuralNetworkRb::MNIST.test_set
      @training_cnt = @training_set.data.shape[0]
    end

    context 'use sigmoid' do

      let(:rack) {  
        NeuralNetworkRb::NeuralNetwork::Builder.new do 
          neuron_count = 100
          learning_rate = 0.001
          use NeuralNetworkRb::Layer::Dot, width: neuron_count, name: :dot1, learning_rate: learning_rate
          use NeuralNetworkRb::Layer::Sigmoid
          use NeuralNetworkRb::Layer::Dot, width: 10, name: :dot2, learning_rate: learning_rate
          use NeuralNetworkRb::Layer::SoftmaxCrossEntropy
          use NeuralNetworkRb::Layer::CrossEntropyFetch, every: 100 do |epoch, error|
            print "#{epoch} #{error}\r"
            $stdout.flush
          end
        end
      }
  
      it 'runs the training loop' do
        targets = NeuralNetworkRb::MNIST.embed_labels(@training_set.labels, :one_hot, 10)
        inputs  = @training_set.data
        epochs.times do |epoch|
          indexes = Numo::Int32.new(batch_size).rand(0, @training_cnt).to_a
          network.train(inputs[indexes, true], targets[indexes, true]) 
        end
        test_accurancy = NeuralNetworkRb.accuracy(network.predict(@test_set.data), @test_set.labels)
        puts test_accurancy # => 0.9637
      end
  
    end

    context 'use relu' do

      let(:rack) {  
        NeuralNetworkRb::NeuralNetwork::Builder.new do 
          neuron_count = 100
          learning_rate = 0.001
          use NeuralNetworkRb::Layer::Dot, width: neuron_count, name: :dot1, learning_rate: learning_rate
          use NeuralNetworkRb::Layer::ReLU
          use NeuralNetworkRb::Layer::Dot, width: 10, name: :dot2, learning_rate: learning_rate
          use NeuralNetworkRb::Layer::SoftmaxCrossEntropy
          use NeuralNetworkRb::Layer::CrossEntropyFetch, every: 100 do |epoch, error|
            puts "#{epoch} #{error}"
          end
        end
      }
  
      it 'runs the training loop' do
        targets = NeuralNetworkRb::MNIST.embed_labels(@training_set.labels, :one_hot, 10)
        inputs  = @training_set.data
        epochs.times do |epoch|
          indexes = Numo::Int32.new(batch_size).rand(0, @training_cnt).to_a
          network.train(inputs[indexes, true], targets[indexes, true]) 
        end

        test_accurancy = NeuralNetworkRb.accuracy(network.predict(@test_set.data), @test_set.labels)
        puts test_accurancy # => 0.9692 and beyond 60000, it breaks out convergence
      end
  
    end

  end
end

