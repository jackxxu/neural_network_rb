RSpec.describe TensorflowRb::NeuralNetwork do

  describe '::train', focus: true do
    let(:neuron_count)  {3}
    let(:learning_rate) {0.05}
    let(:x) { Numo::DFloat[[0,0], [0,1], [1,0], [1,1]] }
    let(:y) { Numo::DFloat[[0],   [1],   [1],   [0]] }
    let(:network) { TensorflowRb::NeuralNetwork.new(neuron_count, learning_rate)}

    context 'train for 2 epochs' do
      before do        
        network.input = x
        network.target = y
      end

      it 'reduces the error for each step' do 
        error1, error2 = 0, 0
        network.train() {|n| error1 = TensorflowRb.l2error(n.target, n.output)}
        network.train() {|n| error2 = TensorflowRb.l2error(n.target, n.output)}
  
        expect(error1).to be > error2
      end
        
    end

    context 'train for 40000 epochs' do
      let(:learning_rate) {0.03}
      let(:epochs) { 40000 }
      before do        
        network.input = x
        network.target = y
      end

      it 'reduces the error for each step' do 
        epochs.times {network.train() {|x| puts TensorflowRb.l2error(network.target, network.output) if network.epoch%1000 == 0  }}
        expect(TensorflowRb.l2error(network.target, network.output)).to be < 0.65
      end 
    end    
  end


end