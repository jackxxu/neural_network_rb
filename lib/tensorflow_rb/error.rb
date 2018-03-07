module TensorflowRb

  def l2error(v1, v2)
    ((v1 - v2) ** 2).sum
  end

  def l1error(v1, v2)
    (v1 - v2).abs.sum
  end

end