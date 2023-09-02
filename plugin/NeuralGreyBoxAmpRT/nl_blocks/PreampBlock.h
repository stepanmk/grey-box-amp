#pragma once

#include <iostream>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
using namespace Eigen;
using json = nlohmann::json;

namespace AmpModel {

    class Preamp {
    public:
        Preamp(int numUnits);

        void loadWeights(json &weights);
        float forward(float in);
        void processBlock(float *block, int numSamples);

    private:
        int numUnits;

        VectorXf recWeightInput{};
        MatrixXf recWeightHidden{};
        VectorXf recBias{};

        VectorXf linWeight{};
        float linBias = 0.f;

        VectorXf vec{};
        VectorXf cellState{};
        VectorXf hiddenState{};

        void resizeVars();
        void resetStates();
        void loadLinWeights(std::vector<float> &weight, std::vector<float> &bias);
        void loadRecWeights(std::vector<float> &weightInput,
                            std::vector<float> &weightHidden,
                            std::vector<float> &bias);
    };

}