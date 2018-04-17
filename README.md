# Rack-Inspired Neural Network Implementation

Neural network's forward and backward propagation bears a lot of resemblance to [Rack]:

* During forward propagation, each layer does its own calculation based on input and then pass the result to the next layer. 
* During backward propagation. each layer gets the gradient from the previous layer, which is used for two purposes
    1. combines gradient passed from the next layer with the gradient from the current layer before sending it back
    2. for some layers with weights and biases, adjusts its weight and bias. 

With a Rack-inspired syntax of the following, you can construct the network like the following
```ruby 
builder = NeuralNetworkRb::NeuralNetwork::Builder.new do 
            use NeuralNetworkRb::Layer::Dot, width: 15, learning_rate: 0.002
            use NeuralNetworkRb::Layer::Sigmoid
            use NeuralNetworkRb::Layer::Dot, width: 10, learning_rate: 0.002
            use NeuralNetworkRb::Layer::SoftmaxCrossEntropy
            use NeuralNetworkRb::Layer::CrossEntropyFetch, every: 100 do |epoch, error|
                puts "#{epoch} #{error}"
            end
          end
network = builder.to_network                  # get the network
epochs.times { network.train(input, target) } # train the network many times
network.predict(test)                         # predict 
```

```
           +--------+        +-------+        +--------+       +--------+       +---------+
input/tgt  |        |inpt/tgt|       |inpt/tgt|        |inpt/tg|        |inpt/tg|         |
+--------> |        +------> |       +------> |        +-----> |        +-----> |         |
           |  Dot   |        |Sigmoid|        |  Dot   |       | Softmax|       | Cross   |
<----------+        | <------+       | <------+        | <-----+ Cross  | <-----+ Entropy |
 gradt     |        | gradt  |       | gradt  |        | gradt | entropy| gradt | Fetch   |
           |        |        |       |        |        |       |        |       |         |
           +--------+        +-------+        +--------+       +--------+       +---------+
```

The syntax for having more layers is very straightforward:

```ruby 
NeuralNetworkRb::NeuralNetwork::Builder.new do 
  use NeuralNetworkRb::Layer::Dot, width: 15, learning_rate: 0.002
  5.times do 
    use NeuralNetworkRb::Layer::Sigmoid
    use NeuralNetworkRb::Layer::Dot, width: 10, learning_rate: 0.002
  end
  use NeuralNetworkRb::Layer::SoftmaxCrossEntropy
  use NeuralNetworkRb::Layer::CrossEntropyFetch, every: 100 do |epoch, error|
    puts "#{epoch} #{error}"
  end
end
```

[Rack]: https://github.com/rack/rack

## Troubleshooting

if you encounter an error message like the following

```
in `load_library': cannot find MKL/OpenBLAS/ATLAS/BLAS-LAPACK library (RuntimeError)
```
it is because your host doesn't have one of these math libraries installed. `numo-linalg` supports a list of these library and has [a benchmark] for their performance. 

I fixed it on my macOS using the following brew install. 

```
brew install openblas
```

[a benchmark]: https://github.com/ruby-numo/numo-linalg/tree/master/bench

## Design Considerations

1. modular design and reusability
2. easily understandable code & process.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'neural_network_rb'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install neural_network_rb
