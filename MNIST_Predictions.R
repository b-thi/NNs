### Classification Code: MNIST

## Libraries
library(keras)
library(tidyr)
library(ggplot2)

## Loading data
fashion_mnist <- dataset_fashion_mnist()
str(fashion_mnist)
## Organizing data
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
str(train_images)
## Names for labels in a vector
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

## Looking at training images structure
dim(train_images) # 60k images, 28 by 28 pixels so we have 784 input variables

## Looking at labels
dim(train_labels)

### Looking at the images
## Note that the grayscale value varies from 0 to 255

# Wrangling data
image_4 <- as.data.frame(train_images[4, , ])
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

## Scaling data
train_images <- train_images / 255
test_images <- test_images / 255

## Looking at first 25 images
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}

#### Creating model
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>% # Turning image into 784 input variables
  layer_dense(units = 128, activation = 'relu') %>% # 128 neurons with relu activation
  layer_dense(units = 10, activation = 'softmax') # Output layer: 1 of 10 things with softmax
# activation function

## Densely connected means FULLY-CONNECTED (EACH NEURON IS INVOLVED IN THE CALCULATION OF
# EVERY SINGLE NEURON IN THE NEXT LAYER)

## Adding loss function and optimizer
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

## Fitting the model
model %>% fit(train_images, train_labels, epochs = 5)

## Seeing the accuracy
score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

## Making a prediction
predictions <- model %>% predict(test_images)

# Looking at all predictions
predictions[1, ]

# Looking at most likely value
which.max(predictions[1, ])

### Predicting via class instead
class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

## 0 = 1 AND 9 = 10 (labels are 0 based)
