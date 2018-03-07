require "open-uri"

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

    def initialize()
      
    end
  end
end