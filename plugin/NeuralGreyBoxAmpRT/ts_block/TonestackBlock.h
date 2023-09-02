#pragma once

#include "DKModel.h"
#include <Eigen/Core>
#include <juce_audio_processors/juce_audio_processors.h>
using namespace Eigen;


namespace AmpModel {
    class Tonestack {
    public:
        Tonestack(double sampleRate);
        void loadWeights(json &weights, float b, float m, float t);
        void updatePots(float b, float m, float t);
        void processBlock(float *block, int numSamples);
        float processSample(float in);
        void setBass(float b);
        void setMiddle(float m);
        void setTreble(float t);

    private:
        double sampleRate;

        // component values
        static constexpr auto RB = 1000000.0f;
        static constexpr auto RM = 20000.0f;
        static constexpr auto RT = 200000.0f;

        float alphaRB = 1.f;
        float alphaRM = 1.f;
        float alphaRT = 1.f;

        float t1bass = 1.f;
        float t2bass = 1.f;
        float t3bass = 1.f;
        float t4bass = 1.f;

        float t1mid = 1.f;
        float t2mid = 1.f;
        float t3mid = 1.f;
        float t4mid = 1.f;

        float t1treble = 1.f;
        float t2treble = 1.f;
        float t3treble = 1.f;
        float t4treble = 1.f;

        Matrix<float, 3, 3> A;
        Matrix<float, 3, 3> A0;
        Matrix<float, 3, 1> B;
        Matrix<float, 3, 1> B0;
        Matrix<float, 1, 3> D;
        Matrix<float, 1, 3> D0;
        Matrix<float, 1, 1> E;
        Matrix<float, 1, 1> E0;

        Matrix<float, 5, 5> Q;
        Matrix<float, 3, 5> Ux;
        Matrix<float, 1, 5> Uo;
        Matrix<float, 1, 5> Uu;

        Matrix<float, 3, 3> ZGx;
        Matrix<float, 5, 5> RvQ_inverse;

        Vector<float, 3> state;

        DKMethod::DKModel dkmodel = DKMethod::DKModel(sampleRate);
        void createStateSpace();

        juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> bass;
        juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> middle;
        juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> treble;

    };
}