require "open-uri"
require 'zlib'

module TensorflowRb

  class MNIST

    N_FEATURES = 28 * 28
    N_CLASSES = 10
    FILE_NAMES = %w(train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz)
    ROOT_URL = 'http://yann.lecun.com/exdb/mnist/'

    class << self
      def download!
        FILE_NAMES.each do |name|
          url = "#{ROOT_URL}#{name}"
          fetch_file(name, url)
        end
        self.new
      end

      private
        def fetch_file(filePath, url)
          File.open(filePath, "w") do |output|
            IO.copy_stream(open(url), output)
          end
        end
    end

    def training_labels
      labels(FILE_NAMES[1])
    end

    def test_labels
      labels(FILE_NAMES[3])
    end

    def training_images
      images(FILE_NAMES[0])
    end

    def test_images
      images(FILE_NAMES[2])
    end

    private 
      def images(file_name)
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

      def labels(file_name)
        labels = nil
        Zlib::GzipReader.open(file_name) do |f|
          _, @n_labels = f.read(8).unpack('N2')
          labels = f.read(@n_labels).unpack('C*')
        end
        Numo::Int16.cast(labels)
      end

  end
end