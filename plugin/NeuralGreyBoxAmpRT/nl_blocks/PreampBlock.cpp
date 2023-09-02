#include "PreampBlock.h"

namespace AmpModel {

    Preamp::Preamp(int numUnits) :
            numUnits(numUnits) {
        resizeVars();
        resetStates();
    }

    void Preamp::resizeVars() {
        recWeightInput.resize(numUnits * 4);
        recWeightInput.setZero();
        recWeightHidden.resize(numUnits * 4, numUnits);
        recWeightHidden.setZero();
        recBias.resize(numUnits * 4);
        recBias.setZero();

        linWeight.resize(numUnits);
        linWeight.setZero();

        vec.resize(numUnits * 4);
        vec.setZero();
        cellState.resize(numUnits);
        hiddenState.resize(numUnits);
    }

    void Preamp::resetStates() {
        cellState.setZero();
        hiddenState.setZero();
    }

    void Preamp::loadWeights(json &weights) {
        std::vector<float> rwi = weights["preamp"]["recWeightInput"].get<std::vector<float>>();
        std::vector<float> rwh = weights["preamp"]["recWeightHidden"].get<std::vector<float>>();
        std::vector<float> rb = weights["preamp"]["recBias"].get<std::vector<float>>();
        loadRecWeights(rwi, rwh, rb);

        std::vector<float> lw = weights["preamp"]["linWeight"].get<std::vector<float>>();
        std::vector<float> lb = weights["preamp"]["linBias"].get<std::vector<float>>();
        loadLinWeights(lw, lb);
    }

    void Preamp::loadRecWeights(std::vector<float> &weightInput,
                                std::vector<float> &weightHidden,
                                std::vector<float> &bias) {
        assert(weightInput.size() == recWeightInput.size());
        assert(weightHidden.size() == recWeightHidden.size());
        assert(bias.size() == recBias.size());
        for (int i = 0; i < weightInput.size(); ++i) {
            recWeightInput(i) = weightInput[i];
            recBias(i) = bias[i];
        }
        int pos = 0;
        // matrices are saved as col major
        for (int j = 0; j < recWeightHidden.cols(); ++j) {
            for (int i = 0; i < recWeightHidden.rows(); ++i) {
                recWeightHidden(i, j) = weightHidden[pos];
                ++pos;
            }
        }
    }

    void Preamp::loadLinWeights(std::vector<float> &weight, std::vector<float> &bias) {
        assert(weight.size() == linWeight.size());
        for (int i = 0; i < weight.size(); ++i) {
            linWeight(i) = weight[i];
        }
        linBias = bias[0];
    }

    float Preamp::forward(float in) {
        vec = in * recWeightInput + recWeightHidden * hiddenState + recBias;
        cellState = cellState.array() *
                    (1.0f / (1.0f + (-vec.segment(numUnits, numUnits).array()).exp())) +
                    (1.0f / (1.0f + (-vec.segment(0, numUnits).array()).exp())) *
                    (vec.segment(numUnits * 2, numUnits)).array().tanh();
        hiddenState = (1.0f / (1.0f + (-vec.segment(numUnits * 3, numUnits).array()).exp())) * cellState.array().tanh();
        return linWeight.dot(hiddenState) + linBias + in;
    }

    void Preamp::processBlock(float *block, int numSamples) {

        for (int n = 0; n < numSamples; ++n)
        {
            block[n] = forward(block[n]);
        }
    }

} // namespace AmpModel