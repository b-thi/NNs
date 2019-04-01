### Projection Pursuit Regression
# Barinder Thind

# Libraries
library(plotly)
library(tidyverse)

### defining g(v)
g_v <- function(x1, x2) {
  v = (x1 + x2)/2
  gv = 1/(1 + exp(-5*(v - 0.5)))
  return(gv)
}

### generating data

# Initializing variables
x1 = seq(0, 10, 0.01)
x2 = seq(0, 10, 0.01)
gv_matrix = matrix(nrow = length(x1), ncol = length(x2))

# Fillign matrix
for (i in 1:length(x1)) {
  for (j in 1:length(x2)) {
    gv_matrix[i, j] = g_v(x1[i], x2[j])
  }
}

# Plotting surface  
ppr_ex %>% 
  plot_ly(z = ~gv_matrix) %>% 
  add_surface()



test1 = rnorm(10)
mean(test1)
test2 = rnorm(10)
mean(test2)
test3 = test1 + test2
mean(test3)
mean(test1 + test2)

5*0.45
6*0.45
2.7 - 2.25
