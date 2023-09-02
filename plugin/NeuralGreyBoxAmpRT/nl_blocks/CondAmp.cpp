#include "CondAmp.h"

namespace AmpModel {

    CondAmp::CondAmp(int numUnits) :
            numUnits(numUnits) {
        resizeVars();
        resetStates();
    }

    void CondAmp::resizeVars() {
        recWeightInput.resize(numUnits * 4);
        recWeightInput.setZero();

        recWeightBass.resize(numUnits * 4);
        recWeightBass.setZero();
        recWeightMid.resize(numUnits * 4);
        recWeightMid.setZero();
        recWeightTreble.resize(numUnits * 4);
        recWeightTreble.setZero();

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

        condVec.resize(numUnits * 4);
        condVec.setZero();
    }

    void CondAmp::resetStates() {
        cellState.setZero();
        hiddenState.setZero();
    }

    void CondAmp::loadWeights(json &weights, float b, float m, float t) {
        std::vector<float> rwi = weights["condamp"]["recWeightInput"].get<std::vector<float>>();
        std::vector<float> rwb = weights["condamp"]["recWeightBass"].get<std::vector<float>>();
        std::vector<float> rwm = weights["condamp"]["recWeightMid"].get<std::vector<float>>();
        std::vector<float> rwt = weights["condamp"]["recWeightTreble"].get<std::vector<float>>();
        std::vector<float> rwh = weights["condamp"]["recWeightHidden"].get<std::vector<float>>();
        std::vector<float> rb = weights["condamp"]["recBias"].get<std::vector<float>>();
        loadRecWeights(rwi, rwb, rwm, rwt, rwh, rb);

        condVec = recWeightBass * b + recWeightMid * m + recWeightTreble * t;

        std::vector<float> lw = weights["condamp"]["linWeight"].get<std::vector<float>>();
        std::vector<float> lb = weights["condamp"]["linBias"].get<std::vector<float>>();
        loadLinWeights(lw, lb);

        bass.reset(44100, 0.04);
        bass.setCurrentAndTargetValue(b);
        middle.reset(44100, 0.04);
        middle.setCurrentAndTargetValue(m);
        treble.reset(44100, 0.04);
        treble.setCurrentAndTargetValue(t);
    }

    void CondAmp::loadRecWeights(std::vector<float> &weightInput,
                                 std::vector<float> &weightBass,
                                 std::vector<float> &weightMid,
                                 std::vector<float> &weightTreble,
                                 std::vector<float> &weightHidden,
                                 std::vector<float> &bias) {
        assert(weightInput.size() == recWeightInput.size());
        assert(weightBass.size() == recWeightBass.size());
        assert(weightMid.size() == recWeightMid.size());
        assert(weightTreble.size() == recWeightTreble.size());
        assert(weightHidden.size() == recWeightHidden.size());
        assert(bias.size() == recBias.size());
        for (int i = 0; i < weightInput.size(); ++i) {
            recWeightInput(i) = weightInput[i];
            recWeightBass(i) = weightBass[i];
            recWeightMid(i) = weightMid[i];
            recWeightTreble(i) = weightTreble[i];
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

    void CondAmp::loadLinWeights(std::vector<float> &weight, std::vector<float> &bias) {
        assert(weight.size() == linWeight.size());
        for (int i = 0; i < weight.size(); ++i) {
            linWeight(i) = weight[i];
        }
        linBias = bias[0];
    }

    void CondAmp::updateCondVec(float b, float m, float t) {
        condVec = recWeightBass * b + recWeightMid * m + recWeightTreble * t;
    }

    void CondAmp::setBass(float b) {
        bass.setTargetValue(juce::jlimit(0.0f, 1.0f, b));
    }

    void CondAmp::setMiddle(float m) {
        middle.setTargetValue(juce::jlimit(0.0f, 1.0f, m));
    }

    void CondAmp::setTreble(float t) {
        treble.setTargetValue(juce::jlimit(0.0f, 1.0f, t));
    }


    float CondAmp::forward(float in) {
        vec = in * recWeightInput + condVec + recWeightHidden * hiddenState + recBias;
        cellState = cellState.array() *
                    (1.0f / (1.0f + (-vec.segment(numUnits, numUnits).array()).exp())) +
                    (1.0f / (1.0f + (-vec.segment(0, numUnits).array()).exp())) *
                    (vec.segment(numUnits * 2, numUnits)).array().tanh();
        hiddenState = (1.0f / (1.0f + (-vec.segment(numUnits * 3, numUnits).array()).exp())) * cellState.array().tanh();
        return linWeight.dot(hiddenState) + linBias + in;
    }

    void CondAmp::processBlock(float *block, int numSamples) {

        for (int n = 0; n < numSamples; ++n)
        {
            if (bass.isSmoothing() || middle.isSmoothing() || treble.isSmoothing())
            {
                updateCondVec(bass.getNextValue(), middle.getNextValue(), treble.getNextValue());
            }
            block[n] = forward(block[n]);
        }
    }

} // namespace AmpModel