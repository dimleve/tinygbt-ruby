require "test/unit/assertions"
require 'TreeNode'

include Test::Unit::Assertions

class Tree
  # something like Java getters and setters
  attr_accessor :root
  def build(instances, grad, hessian, shrinkage_rate, param)
    assert_equal(instances.length, grad.length, "Lenght of instances does not match length of gradient")
    assert_equal(grad.length, hessian.length, "Lenght of grad does not match length of hessian")
    self.root = TreeNode.new()
    current_depth = 0
    # build a CART starting from root
    self.root.build(instances, grad, hessian, shrinkage_rate, current_depth, param)
  end

  def predict(x)
    # root is a TreeNode
    self.root.predict(x)
  end

end