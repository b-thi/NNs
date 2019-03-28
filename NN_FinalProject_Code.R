### Final NN Code Using Iceberg Dataset

## Libraries
library(RJSONIO)
library(keras)
library(abind)
library(kohonen)
library(tidyr)
library(ggplot2)

## Setting seed
set.seed(1)

## Reading in dataset

# Image Data
x = train %>% lapply(function(x){
  c(x$band_1,x$band_2)
}) %>% unlist %>% array(dim=c(75,75,1604)) %>% aperm(c(3,1,2))
str(x)
# Values for Output
y = kohonen::classvec2classmat(unlist(lapply(train,function(x){x$is_iceberg})))
str(x)

# Training Set list
nums <- sample(1:1604, 1000)

# Organizing
train_iceberg <- x[nums, , ]
train_truth <- y[nums, 2]
test_iceberg <- x[-nums, , ]
test_truth <- y[-nums, 2]

# Class Names
iceberg_name <- c("Not an Iceberg", "An Iceberg")

## Viewing an observation
image_4 <- as.data.frame(train_iceberg[4, , ])
colnames(image_4) <- seq_len(ncol(image_4))
image_4$y <- seq_len(nrow(image_4))
image_4 <- gather(image_4, "x", "value", -y)
image_4$x <- as.integer(image_4$x)

# Plotting image
ggplot(image_4, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_bw() +
  theme(aspect.ratio = 1) +
  xlab("Pixel Numbers [x]") +
  ylab("Pixel Numbers [y]")

## Need to scale data
train_iceberg <- train_iceberg/max(abs(train_iceberg))
test_iceberg <- test_iceberg/max(abs(train_iceberg))

## Looking at the first 25 images
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_iceberg[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:75, 1:75, img, col = gray((-44:0)/-44), xaxt = 'n', yaxt = 'n',
        main = paste(iceberg_name[train_truth[i] + 1]))
}

#### Creating model

# Initialization
iceberg_nn <- keras_model_sequential()

# Adding Layers
iceberg_nn %>%
  layer_flatten(input_shape = c(75, 75)) %>% # Turning image into 784 input variables
  layer_dense(units = 128, activation = 'relu') %>% # 128 neurons with relu activation, HL1
  layer_dense(units = 128, activation = 'relu') %>% # 128 neurons with relu activation, HL2
  layer_dense(units = 128, activation = 'relu') %>% # 128 neurons with relu activation, HL3
  layer_dense(units = 128, activation = 'relu') %>% # 128 neurons with relu activation, HL4
  layer_dense(units = 1, activation = 'softmax') # Output layer: 1 of 10 things with softmax
# activation function

## Densely connected means FULLY-CONNECTED (EACH NEURON IS INVOLVED IN THE CALCULATION OF
# EVERY SINGLE NEURON IN THE NEXT LAYER)

## Adding loss function and optimizer
iceberg_nn %>% compile(
  optimizer = 'sgd', # Using stochastic gradient descent as backprop method
  loss = 'binary_crossentropy', # Using cross-entropy as loss evaluator
  metrics = c('accuracy') # Looking at accuracy
)

## Fitting the model
iceberg_nn %>% fit(train_iceberg, train_truth, epochs = 150)

## Seeing the accuracy
score <- iceberg_nn %>% evaluate(test_iceberg, test_truth)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")
