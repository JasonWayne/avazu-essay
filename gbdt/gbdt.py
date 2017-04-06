import xgboost as xgb
import sys
import time

train_file = sys.argv[1]
test_file = sys.argv[2]

# read in data
dtrain = xgb.DMatrix(train_file)
dtest = xgb.DMatrix(test_file)

# specify parameters via map
param = {'max_depth': 5, 'eta': 0.3, 'silent': 1, 'objective':'binary:logistic' }
num_round = 50
bst = xgb.train(param, dtrain, num_round)
# make prediction
start_time = time.time()
preds = bst.predict(dtest)
end_time = time.time()
predict_time = end_time - start_time
print("predict time is " + str(predict_time) + " seconds!!!!")

