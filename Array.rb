class Array
  def square!
    self.map! {|num| num ** 2}
  end

  def add!(number)
    self.map! {|num| num + number}
  end

  def subtract(other_ary)
    self.map.with_index {|v, i| (v-Float(other_ary[i]))}
  end

  def multiply(other_ary)
    self.map.with_index {|v, i| (v*other_ary[i])}
  end

  def self.rand_array(x, max)
    x.times.map{ Random.rand(max) / 100.0 }
  end

  def sigmoid!()
    """The famous sigmoid function"""
    self.map! {|num| 1.0 / (1.0 + Math.exp(-num))}
  end

  def sum!
    self.inject(0){|sum,x| sum + x }
  end

  def get_dimension a
    return 0 if a.class != Array
    result = 1
    a.each do |sub_a|
      if sub_a.class == Array
        dim = get_dimension(sub_a)
        result = dim + 1 if dim + 1 > result
      end
    end
    return result
  end

  def filter f
    select.with_index { |e,i| f.include?(i) == true }
  end

  def self.fixed_array(size, val)
    Array.new(size) { |i| val }
  end
end

