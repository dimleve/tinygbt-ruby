require 'Array'

class MLMetrics
  def self.log_loss_metric(real, pred)
    sum_log_loss = 0.0
    real.zip(pred).each do |real_label, prediction|
      if real_label == 1.0
        sum_log_loss = sum_log_loss + (-Math.log(prediction))
      else
        sum_log_loss = sum_log_loss + (-Math.log(1 - prediction))
      end
    end
    sum_log_loss / real.length
  end

end

