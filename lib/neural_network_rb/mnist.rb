require "open-uri"
require 'zlib'

module NeuralNetworkRb

  class MNIST

    N_FEATURES = 28 * 28
    N_CLASSES = 10
    TRAIN_FILE_NAMES = %w(train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz)
    TEST_FILE_NAMES = %w(t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz)
    ROOT_URL = 'http://yann.lecun.com/exdb/mnist/'

    class << self

      def training_set
        download(TRAIN_FILE_NAMES)        
      end

      def test_set
        download(TEST_FILE_NAMES)        
      end

      def display_image(narray)
        narray.shape[0].times do |r|
          puts narray[r*28..((r+1)*28-1)].to_a.inspect
        end
      end

      def embed_labels(labels, algorithm, class_count)
        NeuralNetworkRb::Embeddings.send(algorithm, labels, class_count)
      end
  
      private 
        def download(files)
          files.each do |name|
            if !File.exists?(name) 
              url = "#{ROOT_URL}#{name}"
              fetch_file(name, url)
            end
          end
          self.new(*files)
        end

        def fetch_file(filePath, url)
          File.open(filePath, "w") do |output|
            IO.copy_stream(open(url), output)
          end
        end

    end

    attr_accessor :labels, :data, :validation_data, :validation_labels

    def initialize(data_file = nil, label_file = nil)
      @labels = get_labels(label_file) if label_file
      @data   = get_images(data_file)  if data_file
    end

    def clone
      self.class.new.tap do |m|
        m.data = self.data.copy
        m.labels = self.labels.copy
        m.validation_data = self.validation_data.copy if self.validation_data
        m.validation_labels = self.validation_labels.copy if self.validation_labels
      end
    end
    
    def shuffle!(seed)
      @data, @labels = NeuralNetworkRb.shuffle(@data, @labels, seed)
      self
    end

    def partition!(train_ratio)
      @data, @validation_data = NeuralNetworkRb.split(@data, train_ratio)
      @labels, @validation_labels = NeuralNetworkRb.split(@labels, train_ratio)
      self
    end


    def batches(batches_count)
      total_size = self.data.shape[0]
      batch_size = (total_size.to_f/batches_count).ceil
      Array.new(batches_count).tap do |result|
        batches_count.times do |i|
          range = batch_size*i..batch_size*(i+1)-1
          batch_data   = NeuralNetworkRb.rows(@data,   range) # self.data[batch_size*i..batch_size*(i+1)-1, true]
          batch_labels = NeuralNetworkRb.rows(@labels, range) # self.labels[batch_size*i..batch_size*(i+1)-1]
          result[i] = [batch_data, batch_labels]
        end
      end
    end

    private 

      def get_images(file_name)
        images = []
        Zlib::GzipReader.open(file_name) do |f|
          _, n_images = f.read(8).unpack('N2')
          n_rows, n_cols = f.read(8).unpack('N2')
          
          n_images.times do
            images << f.read(n_rows * n_cols).unpack('C*')
          end
        end
        Numo::Int16.cast(images)  
      end

      def get_labels(file_name)
        labels = nil
        Zlib::GzipReader.open(file_name) do |f|
          _, @n_labels = f.read(8).unpack('N2')
          labels = f.read(@n_labels).unpack('C*')
        end
        Numo::Int16.cast(labels)
      end

  end
end