# Rack-Inspired Neural Network Implementation

Neural network's forward and backward propagation bears a lot of resemblance to [Rack]:

* During forward propagation, each layer does its own calculation based on input and then pass the result to the next layer. 
* During backward propagation. each layer gets the gradient from the previous layer, which is used for two purposes
    1. combines gradient passed from the next layer with the gradient from the current layer before sending it back
    2. for some layers with weights and biases, adjusts its weight and bias. 

With a Rack-inspired syntax of the following, you can construct the network like the following
```ruby 
NeuralNetworkRb::NeuralNetwork::Builder.new do 
    use NeuralNetworkRb::Activations::Dot, width: 15, learning_rate: 0.002
    use NeuralNetworkRb::Activations::Sigmoid
    use NeuralNetworkRb::Activations::Dot, width: 10, learning_rate: 0.002
    use NeuralNetworkRb::Loss::SoftmaxCrossEntropy
    use NeuralNetworkRb::Loss::CrossEntropyFetch, every: 100 do |epoch, error|
        puts "#{epoch} #{error}"
    end
end
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
    use NeuralNetworkRb::Activations::Dot, width: 15, learning_rate: 0.002
    5.times do 
        use NeuralNetworkRb::Activations::Sigmoid
        use NeuralNetworkRb::Activations::Dot, width: 10, learning_rate: 0.002
    end
    use NeuralNetworkRb::Loss::SoftmaxCrossEntropy
    use NeuralNetworkRb::Loss::CrossEntropyFetch, every: 100 do |epoch, error|
        puts "#{epoch} #{error}"
    end
end
```

[Rack]: https://github.com/rack/rack

## Design Considerations

1. modular design and reusability
2. easy to understand the code and the process.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'neural_network_rb'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install neural_network_rb
