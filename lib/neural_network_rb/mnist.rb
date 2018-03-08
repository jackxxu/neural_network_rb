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

    def initialize(data_file, label_file)
      data_file = data_file
      label_file = label_file
      @labels = get_labels(label_file)
      @data = get_images(data_file)
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