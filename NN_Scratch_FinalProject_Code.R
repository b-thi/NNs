## Final Project NN From Scratch

### Libraries
# none :3

### Setting seed
set.seed(25)

## Defining data frame
x <- rnorm(100)
y <- ifelse(x >= -0.5 & x <= 0.5, 1, 0)
gaussian_df <- sample(data.frame(rn = x, resp = y))

## Looking at data set
head(gaussian_df)

## Activation Function
sigmoid <- function(x) {
  return(1.0/(1.0 + exp(-x)))
}

## Derivative of the activation
sigmoid_deriv <- function(x) {
  return(x*(1.0 - x))
}

## Loss Function
MSE <- function(neural_net) {
  return(mean((neural_net$y - round(neural_net$output))^2))
}

## Initializing
layer_weights_1 <- c(runif(length(gaussian_df$rn)))
layer_weights_2 <- c(runif(length(gaussian_df$rn)))
layer_bias_1 <- c(runif(length(gaussian_df$rn)))
layer_bias_2 <- c(runif(length(gaussian_df$rn)))

## Setting up neural network list
neuralnet_info <- list(
  input = gaussian_df$rn,
  layer_weights_1 = layer_weights_1,
  layer_bias_1 = layer_bias_1,
  layer_weights_2 = layer_weights_2,
  layer_bias_2 = layer_bias_2,
  y = gaussian_df$resp,
  output = matrix(rep(0, 100), ncol = 1)
)

## Forward pass
forward_pass <- function(neural_net) {
  
  # Layer 1 activations
  neural_net$layer1 <- c(sigmoid(neural_net$input * neural_net$layer_weights_1 + 
                                   layer_bias_1))
  
  # Output activations
  neural_net$output <- c(sigmoid(neural_net$layer1 * neural_net$layer_weights_2 + 
                                   layer_bias_2))
  
  return(neural_net)
}

## Backpropagation
grad_descent <- function(neural_net){
  
  ## Easier derivative first
  # weights closer to the output layer
  deriv_weights2 <- (
    neural_net$layer1*(2*(neural_net$y - neural_net$output)*sigmoid_deriv(neural_net$output))
  )
  
  ## Backpropagating to first layer
  # Applied chain rule here
  deriv_weights1 <- (2*(neural_net$y - neural_net$output)*sigmoid_deriv(neural_net$output))*neural_net$layer_weights_2
  deriv_weights1 <- deriv_weights1*sigmoid_deriv(neural_net$layer1)
  deriv_weights1 <- neural_net$input*deriv_weights1
  
  ## Now need to do bias derivatives
  deriv_bias2 <- 2*(neural_net$y - neural_net$output)*sigmoid_deriv(neural_net$output)
  deriv_bias1 <- 2*(neural_net$y - neural_net$output)*sigmoid_deriv(neural_net$output)*layer_weights_2*sigmoid_deriv(neural_net$layer1)
  
  # Weight update using derivative
  learn_rate = 1
  neural_net$layer_weights_1 <- neural_net$layer_weights_1 + learn_rate*deriv_weights1
  neural_net$layer_weights_2 <- neural_net$layer_weights_2 + learn_rate*deriv_weights2
  neural_net$layer_bias_1 <- neural_net$layer_bias_1 + learn_rate*deriv_bias1
  neural_net$layer_bias_2 <- neural_net$layer_bias_2 + learn_rate*deriv_bias2
  
  # Returning updated information
  return(neural_net)
  
}

## Error Rate after no iterations
mean(round(neuralnet_info$output) == gaussian_df$resp)

## Epochs
epoch_num <- 50

## Initializing loss vector
lossData <- data.frame(epoch = 1:epoch_num, MSE = rep(0, epoch_num))

## Training Neural Net
for (i in 1:epoch_num) {
  
  # Foreward iteration
  neuralnet_info <- forward_pass(neuralnet_info)
  
  # Backward iteration
  neuralnet_info <- grad_descent(neuralnet_info)
  
  # Storing loss
  lossData$MSE[i] <- MSE(neuralnet_info)
  
}

## Error Rate after 50 iterations
mean(round(neuralnet_info$output) == gaussian_df$resp)

## Plotting Loss

## I lied, one library
library(tidyverse)

lossData %>% 
  ggplot(aes(x = epoch, y = MSE)) + 
  geom_line(size = 1.25, color = "red") +
  theme_bw() +
  labs(x = "Epoch #", y = "MSE") +
  ggtitle("Change in Loss - Simple Neural Net") +
  theme(plot.title = element_text(hjust = 0.5))

  


