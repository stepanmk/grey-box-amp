#include "TonestackBlock.h"

namespace AmpModel {

    Tonestack::Tonestack(double sampleRate) :
    sampleRate(sampleRate)
    {
    }

    void Tonestack::loadWeights(json &weights, float b, float m, float t) {
        dkmodel.loadCoeffs(weights);
        dkmodel.createComponents(b, m, t);
        dkmodel.createDKModel();

        alphaRB = dkmodel.alphaRB;
        alphaRM = dkmodel.alphaRM;
        alphaRT = dkmodel.alphaRT;

        t1bass = dkmodel.t1bass;
        t2bass = dkmodel.t2bass;
        t3bass = dkmodel.t3bass;
        t4bass = dkmodel.t4bass;
        t1mid = dkmodel.t1mid;
        t2mid = dkmodel.t2mid;
        t3mid = dkmodel.t3mid;
        t4mid = dkmodel.t4mid;
        t1treble = dkmodel.t1treble;
        t2treble = dkmodel.t2treble;
        t3treble = dkmodel.t3treble;
        t4treble = dkmodel.t4treble;

        float bpot = t1bass * std::tanh(t2bass * b + t3bass) + t4bass;
        float mpot = t1mid * std::tanh(t2mid * m + t3mid) + t4mid;
        float tpot = t1treble * std::tanh(t2treble * t + t3treble) + t4treble;

        bass.reset(sampleRate, 0.002);
        bass.setCurrentAndTargetValue(bpot);
        middle.reset(sampleRate, 0.002);
        middle.setCurrentAndTargetValue(mpot);
        treble.reset(sampleRate, 0.002);
        treble.setCurrentAndTargetValue(tpot);

        state.setZero();
        createStateSpace();
    }

    void Tonestack::createStateSpace() {
        Q = dkmodel.Q;
        Ux = dkmodel.Ux;
        Uo = dkmodel.Uo;
        Uu = dkmodel.Uu;

        A0 = dkmodel.A0;
        B0 = dkmodel.B0;
        D0 = dkmodel.D0;
        E0 = dkmodel.E0;

        ZGx = dkmodel.ZGx;
        RvQ_inverse = dkmodel.RvQ_inverse;
        A = A0 - ZGx * Ux * RvQ_inverse * Ux.transpose();
        B = B0 - ZGx * Ux * RvQ_inverse * Uu.transpose();
        D = D0 - Uo * RvQ_inverse * Ux.transpose();
        E = E0 - Uo * RvQ_inverse * Uu.transpose();
    }

    void Tonestack::updatePots(float b, float m, float t) {
        Matrix<float, 5, 5> Rv;
        Rv.setZero();
        Rv(0, 0) = RT * alphaRT * (1 - t);
        Rv(1, 1) = RT * alphaRT * t;
        Rv(2, 2) = RM * alphaRM * (2 * (1 - m)) / (2 - (1 - m));
        Rv(3, 3) = RM * alphaRM * (2 * m) / (2 - m);
        Rv(4, 4) = RB * alphaRB * b;
        RvQ_inverse = (Rv + Q).inverse();
        A = A0 - ZGx * Ux * RvQ_inverse * Ux.transpose();
        B = B0 - ZGx * Ux * RvQ_inverse * Uu.transpose();
        D = D0 - Uo * RvQ_inverse * Ux.transpose();
        E = E0 - Uo * RvQ_inverse * Uu.transpose();
    }

    void Tonestack::setBass(float b) {
        float bpot = t1bass * std::tanh(t2bass * b + t3bass) + t4bass;
        bass.setTargetValue(juce::jlimit(0.0f, 1.0f, bpot));
    }

    void Tonestack::setMiddle(float m) {
        float mpot = t1mid * std::tanh(t2mid * m + t3mid) + t4mid;
        middle.setTargetValue(juce::jlimit(0.0f, 1.0f, mpot));
    }

    void Tonestack::setTreble(float t) {
        float tpot = t1treble * std::tanh(t2treble * t + t3treble) + t4treble;
        treble.setTargetValue(juce::jlimit(0.0f, 1.0f, tpot));
    }

    float Tonestack::processSample(float in)
    {
        float out = (D * state + E * in)(0);
        state = A * state + B * in;
        return out;
    }

    void Tonestack::processBlock(float *block, int numSamples) {

        for (int n = 0; n < numSamples; ++n)
        {
            if (bass.isSmoothing() || middle.isSmoothing() || treble.isSmoothing())
            {
                updatePots(bass.getNextValue(), middle.getNextValue(), treble.getNextValue());
            }
            block[n] = processSample(block[n]);
        }
    }

}