# Multi Layer Feed Forward with Back propagation including momentum
Implementing multi-layer feed forward neural network and training them with backpropagation with momentum with user specific layers and neurons in each layer. Also built a network that will recognize the digits from given MNIST dataset with good accuracy and done tuning with hyper parameters. Also evaluated the performance with confusion matrix and some plots.

#### System Description:
Given dataset contains 5000 records and distribution of each digit in dataset is as shown in table below. Data has been split into train and test dataset in such a way that each class/digit will have distribution in the ratio of 80:20 i.e., train-4000, test-1000 points. Also, again train data has been split to make validation dataset in such a way that train_new-3500 and validate-500. The
distribution each digit into these groups can be seen below. The reason for this type of split is so that network will not be bias to any particular digit.

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
