RSpec.describe TensorflowRb do

  describe 'TensorflowRb.softmax' do    
    let(:data) { Numo::DFloat[[2, 4, 6, 8]] }
    let(:result) { TensorflowRb.softmax(data) }

    it 'calculate the tensorflow output' do
      expect(result).to eq(Numo::DFloat[[ 0.002144008783584634, 0.015842201178506925 , 0.11705891323853292, 0.8649548767993755]])
    end
  end

end
