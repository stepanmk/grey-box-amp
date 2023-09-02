#pragma once
#include "Component.h"
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace DKMethod {

    class DKModel {
    public:
        DKModel(double sampleRate);

        void loadCoeffs(json &coeffs);
        void createComponents(float b, float m, float t);
        void createDKModel();

        std::vector<Component> components;

        Matrix<float, 3, 3> A0;
        Matrix<float, 3, 1> B0;
        Matrix<float, 1, 3> D0;
        Matrix<float, 1, 1> E0;
        Matrix<float, 5, 5> Q;
        Matrix<float, 3, 5> Ux;
        Matrix<float, 1, 5> Uo;
        Matrix<float, 1, 5> Uu;
        Matrix<float, 3, 3> ZGx;
        Matrix<float, 5, 5> RvQ_inverse;

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

    private:

        double sampleRate;
        float alphaR1 = 1.f;
        float alphaR2 = 1.f;
        float alphaC1 = 1.f;
        float alphaC2 = 1.f;
        float alphaC3 = 1.f;

        static constexpr auto res = 5;
        static constexpr auto vres = 5;
        static constexpr auto caps = 3;
        static constexpr auto inputs = 1;
        static constexpr auto outputs = 1;
        static constexpr auto nodes = 8;

        static constexpr auto R1 = 33000.f;
        static constexpr auto R2 = 39000.f;
        static constexpr auto RB = 1000000.f;
        static constexpr auto RM = 20000.f;
        static constexpr auto RT = 200000.f;
        static constexpr auto RV = 1000000.f;

        static constexpr auto C1 = 470e-12f;
        static constexpr auto C2 = 22e-9f;
        static constexpr auto C3 = 22e-9f;

    };

}