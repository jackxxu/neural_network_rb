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

  # describe 'train' do
  #   let(:embedding)     { :one_hot }
  #   let(:label_classes) { 10 }
  #   let(:epochs)        { 10000 }
  #   let(:neuron_count)  { 40 }
  #   let(:learning_rate) { 0.01 }
  #   let(:batches)       { 9 }
  #   let(:random_seed)   { 4567 }
  #   before do
  #     @training_set = NeuralNetworkRb::MNIST.training_set
  #                                           .shuffle!(random_seed)
  #                                           .partition!(0.9)
  #     @test_set     = NeuralNetworkRb::MNIST.test_set
  #     @network = NeuralNetworkRb::NeuralNetwork.new(neuron_count, learning_rate, random_seed)
  #   end

  #   it 'runs the training loop' do
  #     data, labels = *(@training_set.batches(batches)[0])
  #     @network.input = data
  #     @network.target = NeuralNetworkRb::MNIST.embed_labels(labels, :one_hot, label_classes)

  #     epochs.times do 
  #       @network.fit() do |n| 
  #         if (n.epoch % 10 == 1) 
  #           test_accurancy = NeuralNetworkRb.accuracy(@network.predict(@test_set.data), @test_set.labels)
  #           entropy = NeuralNetworkRb.cross_entropy(n.output, n.target)
  #           puts "#{n.epoch}: entropy: #{entropy} #{NeuralNetworkRb.accuracy(n.output, labels)} test: #{test_accurancy}"  
  #         end
  #       end
  #     end 

  #   end
  # end

  describe 'train with rack', focus: true do
    let(:embedding)     { :one_hot }
    let(:epochs)        { 10000 }
    let(:learning_rate) { 0.01 }
    let(:batches)       { 90 }
    let(:random_seed)   { 4567 }

    let(:rack) {  
      NeuralNetworkRb::NeuralNetwork::Builder.new do 
        neuron_count = 100
        use NeuralNetworkRb::Activations::Dot, width: neuron_count, name: :dot1, learning_rate: 0.02
        use NeuralNetworkRb::Activations::Sigmoid
        use NeuralNetworkRb::Activations::Dot, width: 10, name: :dot2, learning_rate: 0.02
        use NeuralNetworkRb::Loss::SoftmaxCrossEntropy
        use NeuralNetworkRb::Loss::CrossEntropyFetch, every: 100 do |epoch, error|
          puts "#{epoch} #{error}"
        end
      end
    }
    let!(:network) { rack.to_network }
    
    before do
      @training_set = NeuralNetworkRb::MNIST.training_set
                                            .shuffle!(random_seed)
                                            .partition!(0.9)
      @test_set     = NeuralNetworkRb::MNIST.test_set
    end

    it 'runs the training loop' do
      input, labels = *(@training_set.batches(batches)[0])
      target = NeuralNetworkRb::MNIST.embed_labels(labels, :one_hot, 10)
      epochs.times { network.train(input, target) }
      test_accurancy = NeuralNetworkRb.accuracy(network.predict(@test_set.data), @test_set.labels)
      puts test_accurancy
    end
  end
end

