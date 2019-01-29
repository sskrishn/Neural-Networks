# Multi Layer Feed Forward with Back propagation including momentum
Implementing multi-layer feed forward neural network and training them with backpropagation with momentum with user specific layers and neurons in each layer. Also built a network that will recognize the digits from given MNIST dataset with good accuracy and done tuning with hyper parameters. Also evaluated the performance with confusion matrix and some plots.

#### System Description:
Given dataset contains 5000 records and distribution of each digit in dataset is as shown in table below. Data has been split into train and test dataset in such a way that each class/digit will have distribution in the ratio of 80:20 i.e., train-4000, test-1000 points. Also, again train data has been split to make validation dataset in such a way that train_new-3500 and validate-500. The
distribution each digit into these groups can be seen below. The reason for this type of split is so that network will not be bias to any particular digit.
![Data Img](https://github.com/sskrishn/Neural-Networks/blob/master/Multi%20Layer%20Feed%20Forwad%20and%20Auto%20Encoder/dataSummary.PNG)
I have written code such that it accepts any number of layers, any number of neurons in each layer, any activation function (from tanh, relu, sigmoid) corresponding to layers, learning rate, momentum, subset of train size, and a flag to run for MLP and Auto Encoder)

#### Neurons in Hidden Layer: 
I have made choice of neurons 100, 150, 200 in hidden layer and observed that with increase in hidden neurons is increasing time taken for each epoch but with almost similar hit rate only difference in 0.001 terms. Also tried keeping 10 neurons and observed that hit rate is very bad for few epochs and so eliminated that case. So I am considering hidden neurons as 150 considering both time and accuracy. Also tried putting more layers which didn’t gave good results.

#### Activation Functions: 
Tried with different combinations – (sigmoid, sigmoid), (relu, relu) and (relu, sigmoid). I have not considered tanh in combination set as the data range is between 0 and 1 and not between -1 and 1. (Relu, Relu) combination is giving good hit rate in less epochs and so considered it for final network.

#### Weights Initialization: 
First I have tried with random numbers between 0 and 1 which are not giving good results as it is taking more epochs to get significant change in hit rate. So I have chosen weights which are giving good results from intial epochs in such a way that they are
Uniform between (-a, +a) where a = √(6/(𝑁_𝑠+𝑁_𝑡 ))
N_s = number of neurons in source layer, Nt = number of neurons in target layer
*Xavier Glorot, Yoshua Bengio, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics,
PMLR 9:249-256, 2010.

#### Leaning rate:
With decreasing learning rate, I have observed that training time is taking more time to get good hit rate i.e., slowly learning takes place. Also I have taken very high learning rate which is giving me very low hit rate and so I have made my choice as 0.01 

#### Momentum: 
I have tried without giving momentum at first, and observed that to get decent rate it took more epochs. When I used in momentum, there is decent hit rate with less epochs. This is due to that momentum is considering the previous weights as weightage in calculating error. I have tried with 0.3 and 0.7 and got good hit rate with less epochs with m=0.7

#### Rule for Stopping:
Initially, I run without condition for 500 epochs and observed that validation and train hit rates are increasing and approaching towards 1. But after some epochs validation hit rate starts fall and again raise for next few epochs and then observed for certain
period it is constant. I have taken condition that when validation error continuously increasing for next 3 tens of epochs as we are calculating hit rate for every 10th epoch to stop the loop.

#### Choice of Ranges: 
Very beginning I tried with random weights and ranges as 0.75 and 0.25 while calculating hit rate and I am getting some points with more than one element less than 0.25 and so unable to assign label and so I used max threshold concept for evaluation of training
set’s hit rate which is in sync with evaluation of test set and validation set.

#### Conclusion :
From the results we can say that when Multi-layer feed forward neural network is trained with MNIST data, if we consider more hidden layers the accuracy is not that good and so we decreased the hidden layers to get decent accuracy. Also we have observed the tuning of parameters will lead to get good accuracy as explained in system description section. Also in any case the network is getting saturated after few hundreds of epochs and cannot learn more than that until and unless we change any parameter. We got testing hit rate of 0.93 training hit rate of 0.982 and validation hit rate of 0.928 within 100 epochs with lr=0.01, m=0.7,150 hidden neurons and activation
function as relu-relu for hidden and output layer. Also from confusion matrix we can say that 5’s are almost classified as correctly and 0’1 and 1’s are also almost classified correctly and 8 has more misclassification when compared with other digits. This behavior may be due to some samples of some digits are more in dataset and so trained accordingly. Also I have plotted the results at the end of document when changing parameters and observed conclusions which are described in system description section.

# Auto Encoder

To train auto-encoder network using the same data as in above problem and to obtain good set of features for representing them. Also evaluating the performance using J2 loss function to quantify the error. Also need to plots for cost for both training and test set and to plot features of hidden neurons in hidden layer.

#### Neurons in Hidden Layer: 
I am using same number of hidden neurons here in training the auto encoder also. i.e., 150

#### Learning Rate
When using same learning rate i.e., 0.01, it is taking more time to get low cost and so I have changed my learning rate to 0.1 which is giving minimum cost from initial epochs itself. I thought to increase further like 1, but it is not giving less cost instead more cost and so chosen 0.1 as final.
#### Momentum:
I have tried without giving momentum at first, and observed that to get decent rate it took more epochs. When I used in momentum, there is minimum cost with less epochs. This is due to that momentum is considering the previous weights as weightage in calculating error. I have tried with 0.3 and 0.7 and got minimum cost with less epochs with m=0.7
#### Rule for Stopping: 
Initially, I run without condition for 500 epochs and observed that validation and train cost (error) are decreasing. But after some epochs validation cost starts fall and again raise with very minimal change i.e., in terms of 0.001 for next few epochs and then
observed for certain period it is constant. I have taken condition that when validation error continuously increasing for next 3 tens of epochs or constant for 3 tens of epochs as we are calculating cost for every 10th epoch to stop the loop. 

#### Conclusion
From the results we can say that when Auto Encoder network is trained with MNIST data, we are getting minimum loss function in few epochs and are able to reconstruct the input images by extracting features using weights. Also we have observed the tuning of parameters will lead to get very minimum cost function as explained in system description section. Also in any case the network is getting slow down to react after few hundreds of unless we change any parameter. We got validating cost 493.10, training cost 1434.13, testing cost 1042.55 without normalization within 400 epochs with lr=0.1, m=0.7, 150 hidden neurons and activation function as relu-sigmoid for hidden and output layer. From cost plot of each digit we can say that train data set can be reconstructed easily when compared to test set. Also from each cost of digit we can say that digit 1 can be easily reconstructed as it is having less cost (error) and so can be reconstructed easily. While digit 2 is having highest cost and so cannot be reconstructed easily. Here we observed weights of each neuron in hidden layer are getting reacted some particular feature from the input. From features plot we can say that any digit having high pixel at the pixel where weight map is having high value will react to that pixel. And some weights are learning in the same way
and no use. Also I have plotted the results at the end of document when changing parameters and observed conclusions which are described in system description section.




