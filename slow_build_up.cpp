#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <queue>
#include "CartpoleEnv2.h"
#include <cstdint>
#include <tuple>

using namespace torch;
using namespace std;

std::unique_ptr<torch::optim::Optimizer> optimizer;
// DQN Agent
class DQNAgent {
public:
 
//network parameters
   //torch::optim::Adam optimizer_;  
    DQNAgent(int state_size,int action_size,int hidden_size, float learning_rate)
    //after : class vars are being filled either with input or with other values like "cpu"
        : state_size_(state_size), action_size_(action_size), hidden_size_(hidden_size), 
        learning_rate_(learning_rate), device_("cpu"),  
        q_network(nn::Sequential(
            nn::Linear(nn::LinearOptions(state_size, hidden_size).bias(false)),
            nn::Functional(torch::relu),
            nn::Linear(nn::LinearOptions(hidden_size, 128).bias(false)),
            nn::Functional(torch::relu),
            nn::Linear(nn::LinearOptions(128, 64).bias(false)),
            nn::Functional(torch::relu),
            nn::Linear(nn::LinearOptions(64, action_size).bias(false))
        ))//, target_network(q_network)
        
        {
        optimizer = std::make_unique<torch::optim::Adam>(q_network->parameters(), learning_rate);
    
       q_network->to(device_);
      //auto target_q_network_ = std::make_shared<torch::nn::Sequential>(*q_network);
    
}
//Constructor Ende, Public vars and methods
    //std::unique_ptr<torch::optim::Adam> optimizer;
    int  select_action(torch::Tensor state, float epsilon, CartPoleEnv env);

   torch::nn::Sequential getQNetwork(){
        return q_network;
}
   torch::nn::Sequential gettargetQNetwork(){
        return target_network;
}
    void train(CartPoleEnv& env, int num_episodes, int max_steps, float gamma, float epsilon, int batch_size,
         int replay_memory_size, float epsilon_decay, float epsilon_end, int target_update_frequency_)
    //other possible hyperparameters: learning rate(specified manually in optimizer), target_update_frequency(f.e. 10))
    {
        std::deque<std::tuple<Tensor, int, float, Tensor, int>> replay_memory;
        for (int episode = 1; episode <= num_episodes; episode++) {
            // Reset environment
            env.reset();

            // Reset episode statistics
            float episode_reward = 0.0f;
            int episode_steps = 0;

            // Run episode
            while (episode_steps < max_steps) {
                //std::vector<float> state = env.get_observation_space().to(device_);
                // Get the observation space from the environment
                std::vector<float> obs = env.get_state();

                // Convert the observation space to a torch::Tensor
                auto state = torch::from_blob(obs.data(), {1, static_cast<long>(obs.size())}).to(device_);
                // You can now use the obs_tensor as input to your neural network

                // Choose action
                int action = select_action(state, epsilon, env);
                //std::cout << "Action: " << action << std::endl;
                std::vector<float> next_state;
                float reward;
                int done;
                // Take action and observe next state, reward, and done
                std::tie(next_state, reward, done) = env.step(action);

                  std::cout << "New observation: ";
                        for (float val : next_state) {
                            std::cout << val << " ";
                        }
                      std::cout << "Reward: " << reward << " Done: " << done << std::endl;


                torch::TensorOptions options(torch::kCPU); // or torch::kCUDA if you want to use GPU
                options = options.dtype(torch::kFloat32);
                //auto [next_state, reward, done] = env.step(action);
                torch::Tensor next_state_tensor = torch::from_blob(next_state.data(), {1, static_cast<int64_t>(next_state.size())}, options);

                // Move the tensor to the device
                next_state_tensor = next_state_tensor.to(device_);
                //next_state = next_state.to(device_);
                //Tensor reward_tensor = from_blob(&reward, {1}, TensorOptions().dtype(kFloat32)).to(device_);

                // Store experience in replay memory
                auto experience = std::make_tuple(state.clone(), action, reward, next_state_tensor, done);
                // Print the tuple
                replay_memory.push_back(experience);
                if (replay_memory.size() > replay_memory_size) {
                    //assures that the replay_memory is only of the size of the replay_memory size
                    replay_memory.pop_front();
                }

                // Update episode statistics
                episode_reward += reward;
                episode_steps++;

                if (replay_memory.size() > replay_memory_size) {
                    //assures that the replay_memory is only of the size of the replay_memory size
                    replay_memory.pop_front();
                }

                // Update episode statistics
                episode_reward += reward;
                episode_steps++;
                // Update Q-network
           
                if (replay_memory.size() >= batch_size) {
                    update_q_network(gamma, batch_size, replay_memory);
                }
                // Update target Q-network
                if (episode_steps % target_update_frequency_ == 0) { 
                    update_target_network();
                    
                }
                //maybe go into eval mode with target_q_network

                // Check if episode is done, because thresholds where exceeded
                if (done==1) {
                    break;
                }

                // Update current state
                //state = next_state.clone();
                state= next_state_tensor.clone();
            }

            // Print episode statistics
            std::cout << "Episode " << episode << " - Reward: " << episode_reward << " - Steps: " << episode_steps << std::endl;
           decay_epsilon(epsilon,epsilon_decay, epsilon_end);
        }
    }
    void update_q_network(float gamma, int batch_size, std::deque<std::tuple<Tensor, int, float, Tensor, int>>& replay_memory) {
    // Create batch tensors
    std::vector<Tensor> state_batch;
    std::vector<int> action_batch;
    std::vector<float> reward_batch;
    std::vector<Tensor> next_state_batch;
    std::vector<int> done_batch;
    
    for (int i = 0; i < batch_size; i++) {
        // Get experience from replay memory
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, replay_memory.size() - 1);

        // Get a random index within the range of valid indices
        int random_index = dist(gen);
        auto experience = replay_memory[random_index];

        //print replay memory
     

        //std::cout << std::get<0>(experience);
        // Append experience to corresponding batch vector
        state_batch.push_back(std::get<0>(experience));
        action_batch.push_back(std::get<1>(experience));
        reward_batch.push_back(std::get<2>(experience));
        next_state_batch.push_back(std::get<3>(experience));
        done_batch.push_back(std::get<4>(experience));
    }

    // Convert batch tensors to a single tensor for each batch
    auto state_tensor = torch::cat(state_batch, 0);
    auto action_tensor = torch::from_blob(action_batch.data(), {batch_size, 1}, torch::kInt).to(device_);
    auto reward_tensor = torch::from_blob(reward_batch.data(), {batch_size, 1}, torch::kFloat32).to(device_);
    auto next_state_tensor = torch::cat(next_state_batch, 0);
    //auto testing =torch::from_blob(action_batch.data(), {batch_size, 1}, torch::kInt);
    auto done_tensor = torch::from_blob(done_batch.data(), {batch_size, 1}, torch::kInt).to(device_);
//    auto done_tensor = torch::from_blob(done_batch.data(), {batch_size, 1}, torch::kBool);
    // Compute Q-values for current state-action pairs
    //auto action_tensor_int64 = action_tensor.to(torch::kLong);
    //select the Q-value for the action taken in the current state, resulting in a tensor of shape (batch_size[50], 1)
    //auto q_values = q_network->forward(state_tensor).gather(1, action_tensor_int64);
    // Compute target Q-values for next state-action pairs
    //auto next_q_values = torch::zeros_like(q_values);
    //next_q_values =std::get<0>(target_network->forward(next_state_tensor).max(1)).detach();
    auto q_values = q_network->forward(state_tensor);
    cout << target_network << endl;
    auto next_q_values=target_network->forward(next_state_tensor);
    //bellman equation
    //torch::Tensor target_q_values = reward_tensor + gamma * next_q_values.unsqueeze(1) * (1 - done_tensor);
    auto target_q_values = reward_tensor + gamma * next_q_values.unsqueeze(1) * (1 - done_tensor);
    // Compute loss and backpropagate
    //auto loss = torch::mse_loss(q_values, target_q_values);
    auto loss = torch::mse_loss(q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1), target_q_values.detach());
    
    optimizer->zero_grad();
    loss.backward();
    optimizer->step();
    
}
 void update_target_network() {
    std::shared_ptr<torch::nn::Sequential> q_net_copy = std::make_shared<torch::nn::Sequential>(q_network);

        target_network=*q_net_copy;
    }
void decay_epsilon(float epsilon_, float epsilon_decay_, float epsilon_end_) {
    if (epsilon_ > epsilon_end_) {
        epsilon_ *= epsilon_decay_;
        epsilon_ = std::max(epsilon_, epsilon_end_);
    }
    }
    //int  select_action(torch::Tensor state, float epsilon, CartPoleEnv env);

    private:
        int state_size_;
        int action_size_;
        int hidden_size_;
        float learning_rate_;
        torch::Device device_;
        torch::nn::Sequential q_network; 
        std::shared_ptr<torch::nn::Sequential> q_net_copy = std::make_shared<torch::nn::Sequential>(q_network);
        torch::nn::Sequential target_network=*q_net_copy; 
         
       //torch::optim::Adam optimizer_;


        };

 int DQNAgent::select_action(Tensor state, float epsilon, CartPoleEnv env ) {

    int action=0;
    //epsilon is the probability to explore the action space instead of exploit by generating a random value of 0 or 1
    //and comparing it to epsilon, if this is false then we go over to exploitation
    if (torch::rand({1}).item<float>() < epsilon) {
        // Sample a random  from the action space
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, env.get_action_space() - 1);
        int index = dist(gen);
       
        // Select the action corresponding to the sampled index
        action = index;

    } else {

    Tensor q_values = q_network->forward(state);
    std::cout << "Q_Values:   " << std::endl;
    std::cout << q_values << std::endl;
        float first_value=q_values[0][0].item<float>();
      
        // Select the action with the highest Q-value
      //      cout << q_network ->parameters()[0].grad() << endl;
          action = q_values.argmax(1).item<int>();
        
          if(std::isnan(first_value)){
            std::cout << state << std::endl;
            
        }
    }
    return action;
 }
int main(){
    
    //int int int float, state_size,action_size,hidden_size,learning_rate
    CartPoleEnv env;
//  
    int state_size = env.get_state().size();
    int action_size = env.get_action_space();
    float hidden_size= 16;
    float learning_rate= 0.0001;
    int num_episodes=500;
    int max_steps=200;
    ////importance of future rewards => closer o 1: agent will consider future rewards more important than immediate rewards 
    ////=> long term focused policy, if closer to 0 => more focused on immediate rewards
    float gamma=0.99;
    float epsilon=0.15;
    int batch_size= 50;
    int replay_memory_size= 10000;
    int target_update_frequency=100;
    ////agent will explore less and exploit more earlier in training, 
    ////while a lower value means that the agent will explore more and exploit less earlier in training.
    float epsilon_decay=0.995;
    float epsilon_end=0.01;
    DQNAgent agent(state_size,action_size,hidden_size,learning_rate);
    
    torch::nn::Sequential new_q_net= agent.gettargetQNetwork(); 
    //torch::nn::Sequential q_net= agent.getQNetwork();
    //std::shared_ptr<torch::nn::Sequential> q_net_copy = std::make_shared<torch::nn::Sequential>(q_net);
    //new_q_net=*q_net_copy;
    std::shared_ptr<torch::nn::Sequential> q_net_copy = std::make_shared<torch::nn::Sequential>(agent.getQNetwork());
    
    new_q_net=*q_net_copy;
    cout << new_q_net << endl;
      agent.train(env, num_episodes, max_steps, gamma, epsilon, batch_size, replay_memory_size, 
                epsilon_decay, epsilon_end, target_update_frequency);
}
