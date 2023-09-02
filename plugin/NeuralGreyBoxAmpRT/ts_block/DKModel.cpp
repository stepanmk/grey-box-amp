#include "DKModel.h"


namespace DKMethod {

    DKModel::DKModel(double sampleRate) :
    sampleRate(sampleRate)
    {
    }

    void DKModel::loadCoeffs(json &coeffs) {
        alphaR1 = coeffs["tonestack"]["R1"].get<float>();
        alphaR2 = coeffs["tonestack"]["R2"].get<float>();
        alphaRB = coeffs["tonestack"]["RB"].get<float>();
        alphaRM = coeffs["tonestack"]["RM"].get<float>();
        alphaRT = coeffs["tonestack"]["RT"].get<float>();
        alphaC1 = coeffs["tonestack"]["C1"].get<float>();
        alphaC2 = coeffs["tonestack"]["C2"].get<float>();
        alphaC3 = coeffs["tonestack"]["C3"].get<float>();

        t2bass = coeffs["tonestack"]["basspot"].get<std::vector<float>>()[0];
        t3bass = coeffs["tonestack"]["basspot"].get<std::vector<float>>()[1];
        t2mid = coeffs["tonestack"]["midpot"].get<std::vector<float>>()[0];
        t3mid = coeffs["tonestack"]["midpot"].get<std::vector<float>>()[1];
        t2treble = coeffs["tonestack"]["treblepot"].get<std::vector<float>>()[0];
        t3treble = coeffs["tonestack"]["treblepot"].get<std::vector<float>>()[1];

        t1bass = 1.f / (std::tanh(t2bass + t3bass) - std::tanh(t3bass));
        t4bass = -t1bass * std::tanh(t3bass);
        t1mid = 1.f / (std::tanh(t2mid + t3mid) - std::tanh(t3mid));
        t4mid = -t1mid * std::tanh(t3mid);
        t1treble = 1.f / (std::tanh(t2treble + t3treble) - std::tanh(t3treble));
        t4treble = -t1treble * std::tanh(t3treble);
    }

    void DKModel::createComponents(float b, float m, float t) {
        b = t1bass * std::tanh(t2bass * b + t3bass) + t4bass;
        m = t1mid * std::tanh(t2mid * m + t3mid) + t4mid;
        t = t1treble * std::tanh(t2treble * t + t3treble) + t4treble;
        float alphaB = b;
        float alphaM1 = (2.f * (1.f - m)) / (2.f - (1.f - m));
        float alphaM2 = (2.f * m) / (2.f - m);
        float alphaT1 = 1.f - t;
        float alphaT2 = t;

        components.clear();

        components.emplace_back(Component("R1", "res", {1, 2}, R1 * alphaR1));
        // TREBLE
        components.emplace_back(Component("VR1_1", "vres", {3, 4}, RT * alphaT1 * alphaRT));
        components.emplace_back(Component("VR1_1", "vres", {4, 5}, RT * alphaT2 * alphaRT));
        // MID
        components.emplace_back(Component("VR2_1", "vres", {6, 7}, RM * alphaM1 * alphaRM));
        components.emplace_back(Component("VR2_2", "vres", {7, 0}, RM * alphaM2 * alphaRM));
        components.emplace_back(Component("R2_1", "res", {6, 7}, RM * 2 * alphaRM));
        components.emplace_back(Component("R2_2", "res", {7, 0}, RM * 2 * alphaRM));
        // BASS
        components.emplace_back(Component("VR3", "vres", {5, 6}, RB * alphaB * alphaRB));
        // VOLUME
        components.emplace_back(Component("R2", "res", {4, 8}, R2 * alphaR2));
        components.emplace_back(Component("RV", "res", {8, 0}, RV));
        // caps
        components.emplace_back(Component("C1", "cap", {1, 3}, C1 * alphaC1));
        components.emplace_back(Component("C2", "cap", {2, 5}, C2 * alphaC2));
        components.emplace_back(Component("C3", "cap", {2, 7}, C3 * alphaC3));
        // ports
        components.emplace_back(Component("input", "in", {1, 0}, 0));
        components.emplace_back(Component("output", "out", {8, 0}, 0));
    }

    void DKModel::createDKModel() {
        Matrix<float, res, nodes> Nr;
        Nr.setZero();
        Matrix<float, res, res> Gr;
        Gr.setZero();

        Matrix<float, vres, nodes> Nv;
        Nv.setZero();
        Matrix<float, vres, vres> Rv;
        Rv.setZero();

        Matrix<float, caps, nodes> Nx;
        Nx.setZero();
        Matrix<float, caps, caps> Gx;
        Gx.setZero();
        Matrix<float, caps, caps> Z;
        Z.setZero();

        Matrix<float, inputs, nodes> Nu;
        Nu.setZero();
        Matrix<float, outputs, nodes> No;
        No.setZero();

        int resCount = 0; int vresCount = 0; int capsCount = 0; int inputsCount = 0; int outputsCount = 0;
        for (auto &component : components)
        {
            if (component.type == "res")
            {
                if (component.nodes(0, 0) > 0)
                    Nr(resCount, component.nodes(0, 0) - 1) = 1.f;
                if (component.nodes(0, 1) > 1)
                    Nr(resCount, component.nodes(0, 1) - 1) = -1.f;
                Gr(resCount, resCount) = 1.f / component.value;
                ++resCount;
            }
            else if (component.type == "vres")
            {
                if (component.nodes(0, 0) > 0)
                    Nv(vresCount, component.nodes(0, 0) - 1) = 1.f;
                if (component.nodes(0, 1) > 1)
                    Nv(vresCount, component.nodes(0, 1) - 1) = -1.f;
                Rv(vresCount, vresCount) = component.value;
                ++vresCount;
            }
            else if (component.type == "cap")
            {
                if (component.nodes(0, 0) > 0)
                    Nx(capsCount, component.nodes(0, 0) - 1) = 1.f;
                if (component.nodes(0, 1) > 1)
                    Nx(capsCount, component.nodes(0, 1) - 1) = -1.f;
                Gx(capsCount, capsCount) = 2.f * component.value / (1.f / (float) sampleRate);
                Z(capsCount, capsCount) = 1.f;
                ++capsCount;
            }
            else if (component.type == "in")
            {
                if (component.nodes(0, 0) > 0)
                    Nu(inputsCount, component.nodes(0, 0) - 1) = 1.f;
                if (component.nodes(0, 1) > 1)
                    Nu(inputsCount, component.nodes(0, 1) - 1) = -1.f;
                ++inputsCount;
            }
            else if (component.type == "out")
            {
                if (component.nodes(0, 0) > 0)
                    No(outputsCount, component.nodes(0, 0) - 1) = 1.f;
                if (component.nodes(0, 1) > 1)
                    No(outputsCount, component.nodes(0, 1) - 1) = -1.f;
                ++outputsCount;
            }
        }

        MatrixXf S0_temp(nodes, nodes + inputs);
        S0_temp.setZero();
        S0_temp << Nr.transpose() * Gr * Nr + Nx.transpose() * Gx * Nx, Nu.transpose();

        MatrixXf S0(nodes + inputs, nodes + inputs);
        S0.setZero();
        MatrixXf Nu_extended = MatrixXf::Zero(inputs, nodes + inputs);
        Nu_extended(0, 0) = 1.f;
        // system matrix
        S0 << S0_temp, Nu_extended;
        // inverse
        MatrixXf S0_inverse = S0.inverse();
        // extended incidence matrices
        MatrixXf Nvp = MatrixXf::Zero(vres, nodes + inputs);
        MatrixXf Nxp = MatrixXf::Zero(caps, nodes + inputs);
        MatrixXf Nup = MatrixXf::Zero(nodes + inputs, inputs);
        MatrixXf Nop = MatrixXf::Zero(outputs, nodes + inputs);

        MatrixXf Nv_zeros = MatrixXf::Zero(Nv.rows(), Nu.rows());
        Nvp << Nv, Nv_zeros;

        MatrixXf Nx_zeros = MatrixXf::Zero(Nx.rows(), Nu.rows());
        Nxp << Nx, Nx_zeros;

        MatrixXf Nu_zeros = MatrixXf::Zero(Nu.cols(), Nu.rows());
        MatrixXf Nu_eye = MatrixXf::Identity(Nu.rows(), Nu.rows());
        Nup << Nu_zeros, Nu_eye;

        MatrixXf No_zeros = MatrixXf::Zero(No.rows(), Nu.rows());
        Nop << No, No_zeros;
        //
        Q = Nvp * S0_inverse * Nvp.transpose();
        Ux = Nxp * S0_inverse * Nvp.transpose();
        Uo = Nop * S0_inverse * Nvp.transpose();
        Uu = Nup.transpose() * S0_inverse * Nvp.transpose();
        //
        ZGx = 2 * Z * Gx;
        A0 = ZGx * Nxp * S0_inverse * Nxp.transpose() - Z;
        B0 = ZGx * Nxp * S0_inverse * Nup;
        D0 = Nop * S0_inverse * Nxp.transpose();
        E0 = Nop * S0_inverse * Nup;
        RvQ_inverse = (Rv + Q).inverse();

    }

}