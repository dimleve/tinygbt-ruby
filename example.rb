require 'csv'
require 'Array'
require 'Dataset'
require 'GBT'
require 'matrix'

# Binary classification example

csv_text = File.read('.\\data\\data_banknote_authentication.txt')
csv = CSV.parse(csv_text, :headers => false)

# keep 10% for evaluation
csv_valid = csv.shuffle.sample(csv.length * 0.1)
csv_train = csv - csv_valid


mat = Matrix[ *csv_train ]
y_train = mat.column(4).to_a.map(&:to_i)

temp = csv_train.transpose
temp.delete_at(4)
csv_train = temp.transpose

dataset_train = Dataset.new(csv_train, y_train)
  
mat = Matrix[ *csv_valid ]
y_valid = mat.column(4).to_a.map(&:to_i)

temp = csv_valid.transpose
temp.delete_at(4)
csv_valid = temp.transpose

dataset_valid = Dataset.new(csv_valid, y_valid)

params = {}

puts 'Start training...'
gbt = GBT.new()

gbt.train(params,
  dataset_train,
  num_boost_round=100,
  valid_set=dataset_valid,
  early_stopping_rounds=10,
  objective="binary")

puts 'Start predicting...'

preds = Array.fixed_array(dataset_valid.y().length, 0)

dataset_valid.x().each_with_index do |x, xi|
  pred = gbt.predict(x=x, nil, num_iteration=gbt.best_iteration)
  y = dataset_valid.y()[xi]
  preds[xi]  = pred
end

logloss = MLMetrics.log_loss_metric(dataset_valid.y().to_a, preds.sigmoid!)  


puts 'The LogLoss of valid set is:'  + String(logloss)

# Regression example

csv_train = CSV.read('.\\data\\regression.train', { :col_sep => "\t" })

mat = Matrix[ *csv_train ]
y_train = mat.column(0).to_a.map(&:to_i)

temp = csv_train.transpose
temp.delete_at(0)
csv_train = temp.transpose

dataset_train = Dataset.new(csv_train, y_train)


  
csv_test = CSV.read('.\\data\\regression.test', { :col_sep => "\t" })

mat = Matrix[ *csv_test ]
y_test = mat.column(0).to_a.map(&:to_i)

temp = csv_test.transpose
temp.delete_at(0)
csv_test = temp.transpose

dataset_valid = Dataset.new(csv_test, y_test)

params = {}

puts 'Start training...'
gbt = GBT.new()

gbt.train(params,
  dataset_train,
  num_boost_round=20,
  valid_set=dataset_valid,
  early_stopping_rounds=5,
  objective="regression")

puts 'Start predicting...'

preds = Array.fixed_array(dataset_valid.y().length, 0)

dataset_valid.x().each_with_index do |x, xi|
  pred = gbt.predict(x=x, nil, num_iteration=gbt.best_iteration)
  y = dataset_valid.y()[xi]
  preds[xi]  = pred
end

l2_loss = MLMetrics.l2_loss_metric(dataset_valid.y().to_a, preds.sigmoid!)  

puts 'The L2 of valid set is:'  + String(l2_loss ** 0.5)
