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

      private 
        def download(files)
          files.each do |name|
            url = "#{ROOT_URL}#{name}"
            fetch_file(name, url)
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
    
    def shuffle!
      @data, @labels = NeuralNetworkRb.shuffle(@data, @labels)
      self
    end

    def partition!(train_ratio)
      @data, @validation_data = NeuralNetworkRb.split(@data, train_ratio)
      @labels, @validation_labels = NeuralNetworkRb.split(@labels, train_ratio)
      self
    end

    def embed_labels!(algorithm, class_count)
      self.labels = NeuralNetworkRb::Embeddings.send(:one_hot, @labels, class_count)
      self
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