# CharRNN implementation flavours for Lasagne

The included iPython notebook describes four different CharRNN implementation flavours for the Lasagne framework. The implementations differ in training and sampling procedures, but always use the same neural network architecture and the same learning objective.

For this particular CharRNN I used the following very simple network layout:

	* InputLayer (65 input dimensions)
	* LSTMLayer  (50 hidden dimensions)
	* DenseLayer (65 output dimensions)
  
The network is deliberately chosen to be very simple, as I wanted to illustrate the different implementation flavours and I did not want to complicate the discussion by using a deeper network layout.
Of course all examples in the notebook can be extended to deeper layouts.
