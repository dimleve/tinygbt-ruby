require 'Array'

class TreeNode
  # something like Java getters and setters
  attr_accessor :is_leaf, :left_child, :right_child, :split_feature_id, :split_val, :weight
  def _calc_split_gain(grad, hess, grad_l, hess_l, grad_r, hess_r, lambd)
    calc_term(grad_l, hess_l, lambd) + calc_term(grad_r, hess_r, lambd) - calc_term(grad, hess, lambd)
  end

  def calc_term(g, h, lambd)
    sqrt_g = g  ** 2
    plus_lambd = h + lambd
    sqrt_g / plus_lambd
  end

  def calc_term_v2(g, h, lambd)
    sqrt_g  = g.square!
    plus_lambd = h.add!(lambd)
    sqrt_g.zip(plus_lambd).map{|x, y| x / y}
  end

  def _calc_leaf_weight(grad, hessian, lambd)
    """
    Calculate the optimal weight of this leaf node.
    (Refer to Eq5 of Reference[1])
    """
    grad_sum  = grad.sum!()
    hess_sum = hessian.sum!()
    - (grad_sum/ (hess_sum + lambd))
  end

  def build(instances, grad, hessian, shrinkage_rate, depth, param)
    """
    Exact Greedy Alogirithm for Split Finidng
    (Refer to Algorithm1 of Reference[1])
    """

    if depth > param['max_depth']
      self.is_leaf = true
      self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
      return
    end

    grad_sum = grad.sum!()
    hess_sum = hessian.sum!()
    best_gain = 0.0
    best_feature_id = -1
    best_val = 0.0
    best_left_instance_ids = -1
    best_right_instance_ids = -1

    num_features = 0
    if instances.nil? || instances.empty?
      puts "Passed empty instances"
    else
      num_features = instances.to_a[0].length
    end

    # for all feature ids (indexes)
    (0..num_features-1).step(1) do |feature_id|

      g_l = 0.0
      h_l = 0.0

      sorted_instance_ids = instances.map {|row| row[feature_id]}.map.with_index.sort.map(&:last)

      # try to find the best split_value for the current feature id
      (0..sorted_instance_ids.length-1).step(1) do |j|
        # calculate gradient and hessian left
        g_l = g_l + grad[sorted_instance_ids[j]]
        h_l = h_l + hessian[sorted_instance_ids[j]]

        # calculate gradient and hessian right
        g_r = grad_sum - g_l
        h_r = hess_sum - h_l

        # calculate gain when splitting value is instances[sorted_instance_ids[j]][feature_id]
        current_gain = _calc_split_gain(grad_sum, hess_sum, g_l, h_l, g_r, h_r, param['lambda'])

        if current_gain > best_gain
          best_gain = current_gain
          best_feature_id = feature_id
          best_val = instances[sorted_instance_ids[j]][feature_id]
          best_left_instance_ids = sorted_instance_ids[0..j]
          best_right_instance_ids = sorted_instance_ids[j+1..sorted_instance_ids.length]
        end
      end
    end

    if best_gain < param['min_split_gain']
      self.is_leaf = true
      self.weight = _calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
    else
      self.split_feature_id = best_feature_id
      self.split_val = best_val

      self.left_child = TreeNode.new()

      self.left_child.build(instances.filter(best_left_instance_ids),
      grad.filter(best_left_instance_ids),
      hessian.filter(best_left_instance_ids),
      shrinkage_rate,
      depth+1, param)

      self.right_child = TreeNode.new()

      self.right_child.build(instances.filter(best_right_instance_ids),
      grad.filter(best_right_instance_ids),
      hessian.filter(best_right_instance_ids),
      shrinkage_rate,
      depth+1, param)
    end
  end

  def predict(x)
    if self.is_leaf
      self.weight
    else
      if x[self.split_feature_id] <= self.split_val
        self.left_child.predict(x)
      else
        self.right_child.predict(x)
      end
    end
  end

end

