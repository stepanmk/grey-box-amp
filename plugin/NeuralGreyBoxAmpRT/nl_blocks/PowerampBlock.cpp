#include "PowerampBlock.h"

namespace AmpModel {

    Poweramp::Poweramp(int numUnits) :
            numUnits(numUnits) {
        resizeVars();
        resetStates();
    }

    void Poweramp::resizeVars() {
        recWeightInput.resize(numUnits * 3);
        recWeightInput.setZero();
        recWeightHidden.resize(numUnits * 3, numUnits);
        recWeightHidden.setZero();
        recBiasInput.resize(numUnits * 3);
        recBiasInput.setZero();
        recBiasHidden.resize(numUnits * 3);
        recBiasHidden.setZero();

        linWeight.resize(numUnits);
        linWeight.setZero();

        vecInput.resize(numUnits * 3);
        vecInput.setZero();
        vecHidden.resize(numUnits * 3);
        vecHidden.setZero();
        rz.resize(numUnits * 2);
        rz.setZero();

        nvec.resize(numUnits);
        nvec.setZero();

        hiddenState.resize(numUnits);
    }

    void Poweramp::resetStates() {
        hiddenState.setZero();
    }

    void Poweramp::loadWeights(json &weights) {
        std::vector<float> rwi = weights["poweramp"]["recWeightInput"].get<std::vector<float>>();
        std::vector<float> rwh = weights["poweramp"]["recWeightHidden"].get<std::vector<float>>();
        std::vector<float> rbi = weights["poweramp"]["recBiasInput"].get<std::vector<float>>();
        std::vector<float> rbh = weights["poweramp"]["recBiasHidden"].get<std::vector<float>>();
        loadRecWeights(rwi, rwh, rbi, rbh);

        std::vector<float> lw = weights["poweramp"]["linWeight"].get<std::vector<float>>();
        std::vector<float> lb = weights["poweramp"]["linBias"].get<std::vector<float>>();
        loadLinWeights(lw, lb);
    }

    void Poweramp::loadRecWeights(std::vector<float> &weightInput,
                                 std::vector<float> &weightHidden,
                                 std::vector<float> &biasInput,
                                 std::vector<float> &biasHidden) {
        assert(weightInput.size() == recWeightInput.size());
        assert(weightHidden.size() == recWeightHidden.size());
        assert(biasInput.size() == recBiasInput.size());
        assert(biasHidden.size() == recBiasHidden.size());
        for (int i = 0; i < weightInput.size(); ++i) {
            recWeightInput(i) = weightInput[i];
            recBiasInput(i) = biasInput[i];
            recBiasHidden(i) = biasHidden[i];
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

    void Poweramp::loadLinWeights(std::vector<float> &weight, std::vector<float> &bias) {
        assert(weight.size() == linWeight.size());
        for (int i = 0; i < weight.size(); ++i) {
            linWeight(i) = weight[i];
        }
        linBias = bias[0];
    }

    float Poweramp::forward(float in) {
        vecInput = in * recWeightInput + recBiasInput;
        vecHidden = recWeightHidden * hiddenState + recBiasHidden;
        rz = 1.0f / (1.0f + (-(vecInput.segment(0, numUnits * 2).array()
                + vecHidden.segment(0, numUnits * 2).array())).exp());
        nvec = (vecInput.segment(numUnits * 2, numUnits).array()
                + rz.segment(0, numUnits).array() * (vecHidden.segment(numUnits * 2, numUnits).array())).tanh();
        hiddenState = (1.f - rz.segment(numUnits, numUnits).array()) * nvec.array()
                + rz.segment(numUnits, numUnits).array() * hiddenState.array();
        return linWeight.dot(hiddenState) + linBias + in;
    }

    void Poweramp::processBlock(float *block, int numSamples) {

        for (int n = 0; n < numSamples; ++n)
        {
            block[n] = forward(block[n]);
        }
    }

} // namespace AmpModel