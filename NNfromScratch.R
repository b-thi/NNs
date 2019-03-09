#### Neural Net from Scratch
### Barinder Thind

## Libraries
library(tidyverse)

## Functions
neural_net <- function(x, y, hl_neuron_num, iterations) {
  
  # Initializing arrays
  output <- c()
  
  # Creating random weights
  weight <- matrix(data = rnorm(ncol(x)*length(hl_neuron_num)), 
                   nrow = length(hl_neuron_num),
                   ncol = ncol(x))
  
  # Creating random biases
  bias <- rnorm(length(hl_neuron_num))
  
  # Defining sigmoid function
  sigmoid <- function(x){
    return(1/(1 + exp(-x)))
  }
  
  # Defining sigmoid gradient
  sigmoid_deriv <- function(x){
    return(sigmoid(x)*(1 - sigmoid(x)))
  }
  
  # Passing on linear combination
  new_activation1 <- sigmoid(weight%*%x)
  new_activation2 <- sigmoid(new_activation1%*%(rnorm(hl_neuron_num)))
  
  
  
}






