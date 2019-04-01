### RESNET Final Code

## Libraries
library(RJSONIO)
library(keras)
library(abind)
library(kohonen)
library(tidyr)
library(ggplot2)

## Loading data
train = fromJSON("train.json")

# Getting relevant information
x = train %>% lapply(function(x){
  c(x$band_1, 
    x$band_2, 
    apply(cbind(x$band_1,x$band_2), 1, mean))}) %>% 
  unlist %>% 
  array(dim=c(75,75,3,1604)) %>% 
  aperm(c(4,1,2,3))

# Values for Output
y <- classvec2classmat(unlist(lapply (train, function(x) {x$is_iceberg})))

# Training Set list
nums <- sample(1:1604, 1300)

# Organizing
train_iceberg <- x[nums, , , ]
train_truth <- y[nums, ]
test_iceberg <- x[-nums, , , ]
test_truth <- y[-nums, ]

## Prepare model
kernel_size = c(5,5)
input_img = layer_input(shape = c(75, 75, 3), name="img")

## Normalizing data
input_img_norm = input_img %>%
  layer_batch_normalization(momentum = 0.99)

## input CNN
input_CNN = input_img_norm %>%
  layer_conv_2d(32, kernel_size = kernel_size, padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(64, kernel_size = kernel_size,padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_dropout(0.25) 

## first residual
input_CNN_residual = input_CNN %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_conv_2d(128, kernel_size = kernel_size,padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(64, kernel_size = kernel_size,padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu()

input_CNN_residual = layer_add(list(input_CNN_residual,input_CNN))

# ## second residual
input_CNN_residual = input_CNN_residual %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_conv_2d(128, kernel_size = kernel_size,padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(64, kernel_size = kernel_size,padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu()

input_CNN_residual = layer_add(list(input_CNN_residual,input_CNN))

## final CNN
top_CNN = input_CNN_residual %>%
  layer_conv_2d(128, kernel_size = kernel_size,padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_conv_2d(256, kernel_size = kernel_size,padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_dropout(0.25) %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_conv_2d(512, kernel_size = kernel_size,padding = "same") %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_dropout(0.25) %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_global_max_pooling_2d()

## Output layer
outputs = top_CNN %>%
  layer_dense(512,activation = NULL) %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_dropout(0.5) %>%
  layer_dense(256,activation = NULL) %>%
  layer_batch_normalization(momentum = 0.99) %>%
  layer_activation_elu() %>%
  layer_dropout(0.5) %>%
  layer_dense(2,activation = "softmax") ## not sure using softmax is the right thing to do...

## Setting up model
model_resNN <- keras_model(inputs = list(input_img), outputs = list(outputs))

## Setting up functions for model evaluation and passes
model_resNN %>% compile(optimizer = optimizer_adam(lr = 0.001),
                  loss="binary_crossentropy",
                  metrics = c("accuracy"))

## Fitting the model
model_resNN %>% fit(train_iceberg, train_truth, epochs = 150)

## Trying on test data
predictions_resnet <-  mean(round(predict(model_resNN, test_iceberg))[,2] == test_truth[,2])
paste("Test Accuracy (ResNN):", predictions_resnet)
