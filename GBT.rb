require 'Array'
require 'MLMetrics'
require 'Tree'

class GBT
  attr_accessor :params, :best_iteration, :models

  LARGE_NUMBER = 1000000.0
  def initialize()
    self.params = {'gamma' => 0.0,
      'lambda' => 1.0,
      'min_split_gain'  => 0.1,
      'max_depth' => 5,
      'learning_rate' => 0.3,
    }
    self.best_iteration = nil
    self.models = nil
  end

  def _calc_training_data_scores(train_set, models)
    if models.length == 0
      return nil
    end

    x = train_set.x()
    scores = Array.fixed_array(x.length, 0)

    x.each_with_index  { |val,index| scores[index] = self.predict(val, models=models) }

    scores
  end

  def _calc_l2_gradient(train_set, scores)
    labels = train_set.y()
    hessian = Array.fixed_array(labels.length, 2)
    grad = Array.fixed_array(labels.length, 0)

    if scores.nil? || scores.empty?
      grad = Array.rand_array(labels.length, 100)
    else
      for i in (0..labels.legth-1)
        grad[i] = 2 * (scores[i] - labels[i])
      end
    end
    return grad, hessian
  end

  def _calc_gradient(train_set, scores)
    """For now, only L2 loss is supported"""
    _calc_l2_gradient(train_set, scores)
  end

  def _calc_l2_loss(models, data_set)
    sum_errors = 0.0

    data_set.x().each_with_index do |x, xi|
      pred = predict(x, models)
      y = data_set.y()[xi]
      sum_errors = sum_errors + (pred - y) **2
    end

    sum_errors / data_set.x().length
  end

  def _calc_loss(models, data_set)
    """For now, only L2 loss is supported"""
    _calc_l2_loss(models, data_set)
  end

  def logLikelihoodLoss(y_hat, y_true)
    prob = y_hat.sigmoid!

    grad = prob.subtract(y_true)

    fixed_ones = Array.new(y_hat.length) { |i| 1 }
    prob2 = fixed_ones.subtract(prob)
    hess = prob.multiply(prob2)
    return grad, hess
  end

  def _calc_log_loss_gradient(train_set, scores)
    labels = train_set.y()
    hessian = Array.fixed_array(labels.length, 0)
    grad = Array.fixed_array(labels.length, 0)

    if scores.nil? || scores.empty?
      grad = Array.rand_array(labels.length, 100)
      hessian = Array.rand_array(labels.length, 100)
    else
      grad, hessian = logLikelihoodLoss(scores, labels)
    end
    return grad, hessian
  end

  def _calc_logloss_loss(models, data_set)
    preds = Array.fixed_array(data_set.x().length, 0)
    data_set.x().each_with_index do |x, xi|
      preds[xi] = predict(x, models)
    end

    logloss = MLMetrics.log_loss_metric(data_set.y().to_a, preds.sigmoid!)
    return logloss
  end

  def _calc_log_loss(models, data_set)
    _calc_logloss_loss(models, data_set)
  end

  def _build_learner(train_set, grad, hessian, shrinkage_rate)
    learner = Tree.new()
    learner.build(train_set.x, grad, hessian, shrinkage_rate, self.params)
    return learner
  end

  def predict(x, models=nil, num_iteration = nil)
    if models.nil? 
      models = self.models
    end

    if num_iteration.nil?
      num_iteration = models.length
    end

    # assert models is not nil
    sum_preds = 0.0
    for m in (0..num_iteration-1)
      sum_preds = sum_preds + models[m].predict(x)
    end
    sum_preds
  end

  def train(params, train_set, num_boost_round=20, valid_set=None, early_stopping_rounds=5, objective="regression")
    models = []
    shrinkage_rate = 1.0
    best_iteration = nil
    best_val_loss = LARGE_NUMBER
    train_start_time = Time.now

    puts "Training until validation scores don't improve for " + String(early_stopping_rounds) + " rounds."

    for iter_cnt in (0..num_boost_round-1)
      iter_start_time = Time.now
      scores = _calc_training_data_scores(train_set, models)
      if objective == "regression"
        grad, hessian = self._calc_gradient(train_set, scores)
      end
      if objective == "binary"
        grad, hessian = self._calc_log_loss_gradient(train_set, scores)
      end
      
      learner = _build_learner(train_set, grad, hessian, shrinkage_rate)
      
      if iter_cnt > 0
        shrinkage_rate = shrinkage_rate * self.params['learning_rate']
      end
      
      models << learner
      
      if objective == "regression"
        train_loss = self._calc_loss(models, train_set)
        
        if valid_set.nil?
          val_loss = nil
        else
          val_loss = self._calc_loss(models, valid_set)
        end
        
        if val_loss.nil?
          val_loss_str = "-"
        else
          val_loss_str = "#{val_loss.round(10)}"
        end
        
        elapesed = Time.now - iter_start_time
        puts "Iter #{iter_cnt}, Train's L2: #{train_loss.round(10)}, Valid's L2: #{val_loss_str}, Elapsed: #{elapesed.round(2)} secs"
      end
      if objective == "binary"
        train_loss = _calc_log_loss(models, train_set)
        if valid_set.nil?
          val_loss = nil
        else
          val_loss = self._calc_log_loss(models, valid_set)
        end
        if val_loss.nil?
          val_loss_str = "-"
        else
          val_loss_str = "#{val_loss.round(10)}"
        end
        elapesed = Time.now - iter_start_time
        puts "Iter #{iter_cnt}, Train's log loss: #{train_loss.round(10)}, Valid's log loss: #{val_loss_str}, Elapsed: #{elapesed.round(2)} secs"
      end
      
      if !val_loss.nil? and val_loss < best_val_loss
        best_val_loss = val_loss
        best_iteration = iter_cnt
      end
      
      if iter_cnt - best_iteration >= early_stopping_rounds
        puts "Early stopping, best iteration is:"
        puts "Iter #{best_iteration}, Train's loss: #{best_val_loss}"
        break
      end
    end # end for loop
    self.models = models
    self.best_iteration = best_iteration
    puts "Training finished. "
  end

end