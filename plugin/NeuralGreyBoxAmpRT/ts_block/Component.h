#pragma once
#include <Eigen/Dense>
#include <string>

using namespace Eigen;

namespace DKMethod {

    class Component {
    public:
        Component(std::string name,
                  std::string type,
                  std::vector<int> compNodes,
                  float value);

        std::string name;
        std::string type;
        Matrix<int, 2, 2> nodes;
        float value;

    private:
        void setNodes(std::vector<int> &compNodes);

    };

}