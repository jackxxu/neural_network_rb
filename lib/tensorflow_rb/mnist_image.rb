require 'chunky_png'

module TensorflowRb
  class MNISTImage
    def initialize(pixels)
      @pixels = pixels
    end

    def save_to_file(file_path)
      size = Math.sqrt(@pixels.length).to_i
      square = ChunkyPNG::Canvas.new(size, size, ChunkyPNG::Color::TRANSPARENT)
      size.times do |i| 
        size.times do |j|
          square[j, i] = @pixels[i*size + j]
        end
      end
      square.to_image.save(file_path)
    end
  end
end