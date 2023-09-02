#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

using namespace juce;

// borrowed from https://github.com/Thrifleganger/level-meter-demo/
class Meter : public Component
{
public:

    void paint(Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();

        g.setColour(Colours::white);
        g.fillRect(bounds);
        g.setColour(Colours::darkgrey);
        const auto scaledY = jmap(level, -60.f, 0.f, 0.f, static_cast<float>(getHeight()));
        g.fillRect(bounds.removeFromBottom(scaledY));
    }

    void setLevel(const float value) { level = value; }
private:
    float level = -60.f;
};
