#pragma once
#include <iostream>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <juce_audio_processors/juce_audio_processors.h>
using namespace Eigen;
using json = nlohmann::json;

namespace AmpModel {

    class CondAmp {
    public:
        CondAmp(int numUnits);

        void loadWeights(json &weights, float b, float m, float t);
        float forward(float in);
        void processBlock(float *block, int numSamples);
        void updateCondVec(float b, float m, float t);
        void setBass(float b);
        void setMiddle(float m);
        void setTreble(float t);

    private:
        int numUnits;

        VectorXf recWeightInput{};
        VectorXf recWeightBass{};
        VectorXf recWeightMid{};
        VectorXf recWeightTreble{};

        VectorXf condVec{};

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
                            std::vector<float> &weightBass,
                            std::vector<float> &weightMid,
                            std::vector<float> &weightTreble,
                            std::vector<float> &weightHidden,
                            std::vector<float> &bias);

        juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> bass;
        juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> middle;
        juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> treble;
    };

}