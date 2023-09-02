#include "Component.h"

#include <utility>

namespace DKMethod {

    Component::Component(std::string name,
                         std::string type,
                         std::vector<int> compNodes,
                         float value) :
        name(std::move(name)),
        type(std::move(type)),
        value(value)
    {
        nodes.setZero();
        setNodes(compNodes);
    }

    void Component::setNodes(std::vector<int> &compNodes) {
        nodes(0, 0) = compNodes[0];
        nodes(0, 1) = compNodes[1];
    }

}