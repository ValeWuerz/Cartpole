#include "CartpoleEnv2.h"
#include <iostream>
#include <torch/torch.h>

int main() {
    // Create an instance of the CartPoleEnv
    CartPoleEnv env;

    // Get the observation space and action space of the environment
    std::vector<float> observation_space = env.get_observation_space();
    //Tensor observation_space = env.get_observation_space();
    int action_space = env.get_action_space();

    // Print the observation space and action space
    std::cout << "Observation space: ";
    for (float val : observation_space) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    //std::cout << "Action space: ";
    //for (int val : action_space) {
    //    std::cout << val << " ";
    //}
    //std::cout << std::endl;

    // Reset the environment and get the initial observation
    std::vector<float> obs = env.reset();

    // Run a loop that takes random actions in the environment
    for (int i = 0; i < 10000; i++) {
        // Choose a random action
        int action = rand() % 2;
        std::cout << "Action: " << action << std::endl;
        // Take the action in the environment and get the new observation, reward, and done flag
        std::vector<float> new_obs;
        float reward;
        bool done;
        std::tie(new_obs, reward, done) = env.step(action);

        // Print the new observation, reward, and done flag
        std::cout << "New observation: ";
        for (float val : new_obs) {
            std::cout << val << " ";
        }
        std::cout << "Reward: " << reward << " Done: " << done << std::endl;

        // If the episode is done, reset the environment
        if (done) {
            obs = env.reset();
        } else {
            obs = new_obs;
        }
    }

    return 0;
}
