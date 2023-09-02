#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "BinaryData.h"

using namespace juce;

class CustomFontLookAndFeel : public LookAndFeel_V4
{
public:
    CustomFontLookAndFeel()
    {
        LookAndFeel::setDefaultLookAndFeel (this);
    }

    static Font getCustomFont()
    {
        static auto typeface = Typeface::createSystemTypefaceFor (BinaryData::inter_ttf, BinaryData::inter_ttfSize);
        return Font (typeface);
    }

    Typeface::Ptr getTypefaceForFont (const Font& f) override
    {
        return getCustomFont().getTypefacePtr();
    }
private:
};