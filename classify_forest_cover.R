##
# The forest cover problem 
##

## Notes
# Distribution of elevation in test and train sets is different

## load data
load_data <- function() {
  data = read.csv("train.csv")
  data = data[sample(nrow(data)),]
  train.x = data[1:14000,]
  train.y = train.x$Cover_Type
  train.x$Cover_Type = NULL
  train.x$Id = NULL
  test.x = data[14001:15120,]
  test.y = test.x$Cover_Type
  test.x$Cover_Type = NULL
  test.x$Id = NULL
}

## submission
submit <- function() {
  trainall.x = read.csv("train.csv")
  trainall.y = trainall.x$Cover_Type
  trainall.x$Cover_Type = NULL
  trainall.x$Id = NULL
  test = read.csv("test.csv")
  id = test$Id
  test$Id = NULL
}

## naive Bayes
naivebayes <- function() {
  nb = naiveBayes(as.factor(train.y) ~ ., data=train.x)
  nb.pred = predict(nb, newdata=test.x, type="class")
  nb.cm = confusionMatrix(data=nb.pred, reference=test.y)
  print(nb.cm)  ## about .3
}

## decision tree
dt <- function() {
  dtree = rpart(as.factor(train.y) ~ ., data=train.x)
  dtree.pred = predict(dtree, newdata=test.x, type="class")
  dtree.cm = confusionMatrix(data=dtree.pred, reference=test.y)
  print(dtree.cm)  ## about .6
}

## random forest
rforest <- function() {
  rforest.tune = tuneRF(x=data.matrix(train.x), y=as.factor(train.y), ntreeTry=1000)
  rforest = randomForest(as.factor(train.y) ~., data=train.x, ntree=1000,
                         nodesize = 3, mtry=28)
  rforest.pred = predict(rforest, newdata=test.x)
  rforest.cm = confusionMatrix(data=rforest.pred, reference=test.y)
  print(rforest.cm)  ## about .8598
  ##
  ## submit
  ##
  rforest = randomForest(as.factor(trainall.y) ~., data=trainall.x, ntree=1000)
  rforest.pred = predict(rforest, newdata=test)
  rforest.pred = cbind(id, rforest.pred)
  colnames(rforest.pred) = c("Id", "Cover_Type")
  write.csv(rforest.pred, "submission.csv", row.names=FALSE)
}

## logistic regression (lasso regularization)
## takes a while to train...
logreg <- function() {
  logreg = cv.glmnet(x=as.matrix(train.x), y=as.factor(train.y), family="multinomial",
                  alpha=.5)
  logreg.pred = predict(logreg, newx=as.matrix(test.x), type="class",
                        s="lambda.min")
  logreg.cm = confusionMatrix(data=logreg.pred, reference=test.y)
  print(logreg.cm)  ## about .72
}

## neural net
neunet <- function() {
  neunet = nnet(as.factor(train.y) ~., data=train.x, size=8,
                maxit=1500, decay=5e-4)
  neunet.pred = predict(neunet, newdata=test.x, type="class")
  neunet.cm = confusionMatrix(data=neunet.pred, reference=test.y)
  print(neunet.cm)  ## about .65, < 1000 iterations
}

## knn
knn <- function() {
  knn = knn3(as.factor(train.y) ~., data=train.x, k=1)
  knn.pred = predict(knn, newdata=test.x, type="class")
  knn.cm = confusionMatrix(data=knn.pred, reference=test.y)
  print(knn.cm)  ## about .84
  ##
  ## submit
  ##
  knn = knn3(as.factor(trainall.y) ~., data=trainall.x, k=1)
  knn.pred = predict(knn, newdata=test, type="class")
}

## svm
## takes a long time, reaches max iterations
svecm <- function() {
  svecm = svm(as.factor(train.y) ~., data=train.x, kernel="polynomial")
  svecm.pred = predict(svecm, newdata=test.x, type="class")
  svecm.cm = confusionMatrix(data=svecm.pred, reference=test.y)
  print(svecm.cm)  ## about .62 (linear kernel, no tuning)
}

## random ferns
rfern <- function() {
  rfern = rFerns(as.factor(train.y) ~., data=train.x, depth=5)
  rfern.pred = predict(rfern, x=test.x)
  rfern.cm = confusionMatrix(data=rfern.pred, reference=test.y)
  print(rfern.cm)  ## about .6
}

## gbm
grbm <- function() {
  grbm = gbm(as.factor(train.y) ~., data=train.x, n.trees=1500,
             interaction.depth=3, distribution="multinomial",
             shrinkage=.075)
  grbm.pred = predict(grbm, newdata=test.x, n.trees=1500, 
                      type="response")
  grbm.pred = apply(grbm.pred, 1, which.max)
  grbm.cm = confusionMatrix(data=grbm.pred, reference=test.y)
  print(grbm.cm)  
  ## about .8518 for n.trees=3000, shrinkage=.07, interact=2
  ##
  ## submit
  ##
  grbm = gbm(as.factor(trainall.y) ~., data=trainall.x, n.trees=3000,
             interaction.depth=2, distribution="multinomial",
             shrinkage=.075)
  grbm.pred = predict(grbm, newdata=test, n.trees=3000, 
                      type="response")
  grbm.pred = apply(grbm.pred, 1, which.max)
}

## mda
mxda <- function() {
  mxda = mda(as.factor(train.y) ~., data=train.x)
  mxda.pred = predict(mxda, newdata=test.x, type="class")
}

## aggregate neural networks
avnnet <- function() {
  avnnet = avNNet(as.factor(train.y) ~., data=train.x, size=10,
                maxit=1500, decay=5e-2, repeats=10)
  avnnet.pred = predict(avnnet, newdata=test.x, type="class")
  avnnet.cm = confusionMatrix(data=avnnet.pred, reference=test.y)
  print(avnnet.cm)  ## about .7795
}

## multilayer neural net
mnnet <- function() {
  mnnet = monmlp.fit(x=as.matrix(train.x), y=as.matrix(train.y), hidden1=5,
                     hidden2=5)
  mnnet.pred = monmlp.predict(x=as.matrix(test.x), weights=mnnet)
  
}

## extreme random forest
exrf <- function() {
  exrf = extraTrees(x=data.matrix(train.x), y=as.factor(train.y), ntree=200, mtry=10, numRandomCuts=8)
  exrf.pred = predict(exrf, newdata=test.x, type="class")
  exrf.cm = confusionMatrix(data=exrf.pred, reference=test.y)
  print(exrf.cm)  ## about .8911
  ## submit
  exrf = extraTrees(x=data.matrix(trainall.x), y=as.factor(trainall.y), ntree=400, mtry=10,
                    numRandomCuts = 8, numThreads=2) # .79192
  exrf.pred = predict(exrf, newdata=test, type="class")
  exrf.pred = cbind(id, exrf.pred)
  colnames(exrf.pred) = c("Id", "Cover_Type")
  write.csv(exrf.pred, "submission.csv", row.names=FALSE)
}

## deep net
deep <- function() {
  localH2O = h2o.init()
  localH2O = new("H2OClient", ip = "127.0.0.1", port = 54321) 
  train = cbind(train.y,train.x)
  train = as.h2o(localH2O,train,key='train')
  test = as.h2o(localH2O,test.x,key='test')
  deep = h2o.deeplearning(x=2:dim(train)[2],y=1,data=train,
                          activation="Tanh",
                          hidden=c(100,100,100),
                          nesterov_accelerated_gradient=T)
  deep.pred = h2o.predict(deep, test)
  deep.pred = as.data.frame(deep.pred)$predict
  deep.cm = confusionMatrix(data=deep.pred,reference=test.y)
  print(deep.cm)
}

## ctree
ctree <- function() {
  contree = ctree(as.factor(train.y) ~., data=train.x)
  contree.pred = predict(contree, newdata=test.x, type="response")
  contree.cm = confusionMatrix(data=contree.pred, reference=test.y)
  print(contree.cm)  # .7411
}

## cforest
conforest <- function() {
  conforest = cforest(as.factor(train.y) ~., data=train.x, 
                      controls=cforest_control(ntree=1000, mtry=10))
  conforest.pred = predict(conforest, newdata=test.x, type="response")
  conforest.cm = confusionMatrix(data=conforest.pred, reference=test.y)
  print(conforest.cm)
}

## elmNN
elmnn <- function() {
  elmnn = elmtrain(as.factor(train.y) ~., data=train.x,nhid=10,actfun="sig")
  elmnn.pred = predict(elmnn,newdata=test.x,type="response")
  elmnn.cm = confusionMatrix(data=elmnn.pred, reference=test.y)
  print(elmnn.cm)
}