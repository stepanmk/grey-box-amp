#include "AmpModel.h"

namespace AmpModel {
    AmpModel::AmpModel()
    {

    }

    void AmpModel::prepare(json &weights, float b, float m, float t) {
        preamp.loadWeights(weights);
        tonestack.loadWeights(weights, b, m, t);
        poweramp.loadWeights(weights);
    }

    void AmpModel::setBass(float b) {
        tonestack.setBass(b);
    }

    void AmpModel::setMiddle(float m) {
        tonestack.setMiddle(m);
    }

    void AmpModel::setTreble(float t) {
        tonestack.setTreble(t);
    }

    void AmpModel::processBlock(float *block, int numSamples) {
        preamp.processBlock(block, numSamples);
        tonestack.processBlock(block, numSamples);
        poweramp.processBlock(block, numSamples);
    }

}