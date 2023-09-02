#pragma once

//#include <juce_events/juce_events.h>
#include "PluginProcessor.h"
#include "./gui/CustomLNF.h"
#include "./gui/Meter.h"

//==============================================================================
class AudioPluginAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                         public juce::Timer,
                                         private juce::Button::Listener,
                                         private juce::Slider::Listener

{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void timerCallback() override;
    void resized() override;

private:
    void sliderValueChanged(juce::Slider* slider) override;
    void buttonClicked(juce::Button* button) override;

    AudioPluginAudioProcessor& processorRef;

    juce::Slider inGainSlider;
    juce::Label inGainLabel;
    std::unique_ptr <juce::AudioProcessorValueTreeState::SliderAttachment> inGainAttachment;

    juce::Slider outGainSlider;
    juce::Label outGainLabel;
    std::unique_ptr <juce::AudioProcessorValueTreeState::SliderAttachment> outGainAttachment;

    juce::Slider bassSlider;
    juce::Label bassLabel;
    std::unique_ptr <juce::AudioProcessorValueTreeState::SliderAttachment> bassAttachment;

    juce::Slider middleSlider;
    juce::Label middleLabel;
    std::unique_ptr <juce::AudioProcessorValueTreeState::SliderAttachment> middleAttachment;

    juce::Slider trebleSlider;
    juce::Label trebleLabel;
    std::unique_ptr <juce::AudioProcessorValueTreeState::SliderAttachment> trebleAttachment;

    juce::TextButton ts1 {"TS1 Model"};
    juce::TextButton ts3 {"TS3 Model"};
    juce::TextButton rnn21 {"RNN21 Model"};

    CustomFontLookAndFeel lnf;
    Meter inMeterL, inMeterR;
    Meter outMeterL, outMeterR;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};