To make a deep copy of a torch nn:
  std::shared_ptr<torch::nn::Sequential> new_nn_ref = std::make_shared<torch::nn::Sequential>(old_nn);

To access the normal Sequential model, dereferencing is necessary:
 torch::nn::Sequential& new_nn_deref = *new_nn_ref;

No normal Sequential model function can be applied!


Segmentation fault needs to be adressed => solved

batch doesnt store experience of state properly => solved

the q_values are nan after some time beeing normal!
->update_q_network function has a lot of speculation => build knowledge => problem is definitely update_q_network function
=> error is not in replay buffer to batch function, states look good but problem might e in other parameters
Before q_values are calculated, some gradients are nan => why? => check other parameters than state for anomalies
=> weights are -nan first => maybe numerical instability
=>Gradients after loss.backwards() function have nan first => propagates through weights