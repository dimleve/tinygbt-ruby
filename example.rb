require 'csv'
require 'Array'
require 'Dataset'
require 'GBT'
require 'matrix'

def sigmoid(input)
  1.0 / (1.0 + Math.exp(-input))
end

csv_text = File.read('C:\\Users\\dleventis\\tinygbt\\data\\data_banknote_authentication.txt')
csv = CSV.parse(csv_text, :headers => false)

mat = Matrix[ *csv ]
y = mat.column(4).to_a.map(&:to_i)


temp = csv.transpose
temp.delete_at(4)
csv = temp.transpose

dataset = Dataset.new(csv, y)

puts "dataset.x() " + String(dataset.x())
puts "dataset.y() " + String(dataset.y())

params = {}

puts 'Start training...'
gbt = GBT.new()

gbt.train(params,
  dataset,
  num_boost_round=20,
  valid_set=dataset,
  early_stopping_rounds=10,
  objective="binary")

#puts 'Start predicting...'
#y_pred = []
#for x in X_test:
#    y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))
#
#puts 'The LogLoss of prediction is:'  + log_loss(y_test, inverse_logit_function(np.array(y_pred)))
#puts 'The AUC of prediction is:'  + roc_auc_score(y_test, inverse_logit_function(np.array(y_pred)))

