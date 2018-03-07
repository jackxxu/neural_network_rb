RSpec.describe TensorflowRb do

  describe 'l2error' do
    context 'y and y hat are the same' do
      let(:y)     { Numo::DFloat[[0],   [1],   [1],   [0]] }
      let(:y_hat) { Numo::DFloat[[0],   [1],   [1],   [0]] }
      
      it 'produces 0 error' do
        expect(TensorflowRb.l2error(y_hat, y)).to be_within(0.0000001).of(0)
      end
    end

    context 'y and y hat are different' do
      let(:y)     { Numo::DFloat[[0],   [1],   [1],   [0]] }
      let(:y_hat) { Numo::DFloat[[0],   [1],   [1],   [0.5]] }
      
      it 'produces 0 error' do
        expect(TensorflowRb.l2error(y_hat, y)).to be_within(0.0000001).of(0.25)
      end
    end
  end

  describe 'l1error' do
    context 'y and y hat are the same' do
      let(:y)     { Numo::DFloat[[0],   [1],   [1],   [0]] }
      let(:y_hat) { Numo::DFloat[[0],   [1],   [1],   [0]] }
      
      it 'produces 0 error' do
        expect(TensorflowRb.l1error(y_hat, y)).to be_within(0.0000001).of(0)
      end
    end

    context 'y and y hat are different' do
      let(:y)     { Numo::DFloat[[0],   [1],   [1],   [0]] }
      let(:y_hat) { Numo::DFloat[[0],   [1],   [1],   [0.5]] }
      
      it 'produces 0 error' do
        expect(TensorflowRb.l1error(y_hat, y)).to be_within(0.0000001).of(0.5)
      end
    end
    
  end
end

