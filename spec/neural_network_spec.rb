RSpec.describe NeuralNetworkRb::NeuralNetwork do

  # describe '::train' do
  #   let(:neuron_count)  {3}
  #   let(:learning_rate) {0.05}
  #   let(:x) { Numo::DFloat[[0,0], [0,1], [1,0], [1,1]] }
  #   let(:y) { Numo::DFloat[[0],   [1],   [1],   [0]] }
  #   let(:network) { NeuralNetworkRb::NeuralNetwork.new(neuron_count, learning_rate)}

  #   context 'train for 100 epochs' do
  #     before do        
  #       network.input = x
  #       network.target = y
  #     end

  #     it 'reduces the error for each step' do 
  #       error1, error2 = 0, 0
  #       network.fit() {|n| error1 = NeuralNetworkRb.l2error(n.target, n.output)}
  #       100.times { network.fit() }
  #       network.fit() {|n| error2 = NeuralNetworkRb.l2error(n.target, n.output)}
  
  #       expect(error1).to be > error2
  #     end
  #   end

    context 'train for 100000 epochs' do
      let(:learning_rate) {0.05}
      let(:epochs) { 100000 }
      before do        
        network.input = x
        network.target = y
      end

      it 'reduces the error for each step' do 
        epochs.times {network.fit()} 
        expect(NeuralNetworkRb.l2error(network.target, network.output)).to be < 0.65
      end 
    end    
  end


end