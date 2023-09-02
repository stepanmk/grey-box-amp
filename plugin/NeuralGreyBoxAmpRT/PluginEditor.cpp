#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
        : AudioProcessorEditor (&p), processorRef (p)
{

    ts1.setRadioGroupId(1);
    ts1.setButtonText("TS1 Model");
    ts1.setToggleState(true, juce::sendNotificationSync);
    ts1.setMouseCursor(juce::MouseCursor::PointingHandCursor);
    ts1.setColour(juce::TextButton::buttonOnColourId, juce::Colours::darkgrey);
    ts1.setColour(juce::TextButton::buttonColourId, juce::Colours::grey);
    ts1.setClickingTogglesState(true);
    ts1.addListener(this);

    ts3.setRadioGroupId(1);
    ts3.setButtonText("TS3 Model");
    ts3.setMouseCursor(juce::MouseCursor::PointingHandCursor);
    ts3.setColour(juce::TextButton::buttonOnColourId, juce::Colours::darkgrey);
    ts3.setColour(juce::TextButton::buttonColourId, juce::Colours::grey);
    ts3.setClickingTogglesState(true);
    ts3.addListener(this);

    rnn21.setRadioGroupId(1);
    rnn21.setButtonText("RNN21 Model");
    rnn21.setMouseCursor(juce::MouseCursor::PointingHandCursor);
    rnn21.setColour(juce::TextButton::buttonOnColourId, juce::Colours::darkgrey);
    rnn21.setColour(juce::TextButton::buttonColourId, juce::Colours::grey);
    rnn21.setClickingTogglesState(true);
    rnn21.addListener(this);

    addAndMakeVisible(ts1);
    addAndMakeVisible(ts3);
    addAndMakeVisible(rnn21);

    inGainSlider.addListener(this);
    inGainSlider.setRange(0.0f, 1.0f, 0.1f);
    inGainSlider.setValue(0.5f);
    inGainSlider.setSliderStyle(juce::Slider::SliderStyle::RotaryVerticalDrag);
    inGainSlider.setColour(juce::Slider::thumbColourId, juce::Colours::darkgrey);
    inGainSlider.setColour(juce::Slider::rotarySliderFillColourId, juce::Colours::grey);
    inGainSlider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::white);
    inGainSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::darkgrey);
    inGainSlider.setTextBoxStyle(juce::Slider::TextEntryBoxPosition::TextBoxBelow, false, 50, 20);
    inGainSlider.setNumDecimalPlacesToDisplay(1);
    inGainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            processorRef.params, "ingain", inGainSlider);
    addAndMakeVisible(inGainSlider);


    inGainLabel.setText("Input Gain", juce::NotificationType::dontSendNotification);
    inGainLabel.setJustificationType(juce::Justification::centred);
    inGainLabel.setColour(juce::Label::ColourIds::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(inGainLabel);

    outGainSlider.addListener(this);
    outGainSlider.setRange(0.0f, 1.0f, 0.1f);
    outGainSlider.setValue(0.5f);
    outGainSlider.setSliderStyle(juce::Slider::SliderStyle::RotaryVerticalDrag);
    outGainSlider.setColour(juce::Slider::thumbColourId, juce::Colours::darkgrey);
    outGainSlider.setColour(juce::Slider::rotarySliderFillColourId, juce::Colours::grey);
    outGainSlider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::white);
    outGainSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::darkgrey);
    outGainSlider.setTextBoxStyle(juce::Slider::TextEntryBoxPosition::TextBoxBelow, false, 50, 20);
    outGainSlider.setNumDecimalPlacesToDisplay(1);
    outGainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            processorRef.params, "outgain", outGainSlider);
    addAndMakeVisible(outGainSlider);


    outGainLabel.setText("Output Gain", juce::NotificationType::dontSendNotification);
    outGainLabel.setJustificationType(juce::Justification::centred);
    outGainLabel.setColour(juce::Label::ColourIds::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(outGainLabel);

    bassSlider.addListener(this);
    bassSlider.setRange(0.0f, 1.0f, 0.1f);
    bassSlider.setValue(0.5f);
    bassSlider.setSliderStyle(juce::Slider::SliderStyle::RotaryVerticalDrag);
    bassSlider.setColour(juce::Slider::thumbColourId, juce::Colours::darkgrey);
    bassSlider.setColour(juce::Slider::rotarySliderFillColourId, juce::Colours::grey);
    bassSlider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::white);
    bassSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::darkgrey);
    bassSlider.setTextBoxStyle(juce::Slider::TextEntryBoxPosition::TextBoxBelow, false, 50, 20);
    bassSlider.setNumDecimalPlacesToDisplay(1);
    bassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            processorRef.params, "bass", bassSlider);
    addAndMakeVisible(bassSlider);


    bassLabel.setText("Bass", juce::NotificationType::dontSendNotification);
    bassLabel.setJustificationType(juce::Justification::centred);
    bassLabel.setColour(juce::Label::ColourIds::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(bassLabel);

    middleSlider.addListener(this);
    middleSlider.setRange(0.0f, 1.0f, 0.1f);
    middleSlider.setValue(0.5f);
    middleSlider.setSliderStyle(juce::Slider::SliderStyle::RotaryVerticalDrag);
    middleSlider.setColour(juce::Slider::thumbColourId, juce::Colours::darkgrey);
    middleSlider.setColour(juce::Slider::rotarySliderFillColourId, juce::Colours::grey);
    middleSlider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::white);
    middleSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::darkgrey);
    middleSlider.setTextBoxStyle(juce::Slider::TextEntryBoxPosition::TextBoxBelow, false, 50, 20);
    middleSlider.setNumDecimalPlacesToDisplay(1);
    middleAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            processorRef.params, "middle", middleSlider);
    addAndMakeVisible(middleSlider);


    middleLabel.setText("Middle", juce::NotificationType::dontSendNotification);
    middleLabel.setJustificationType(juce::Justification::centred);
    middleLabel.setColour(juce::Label::ColourIds::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(middleLabel);

    trebleSlider.addListener(this);
    trebleSlider.setRange(0.0f, 1.0f, 0.1f);
    trebleSlider.setValue(0.5f);
    trebleSlider.setSliderStyle(juce::Slider::SliderStyle::RotaryVerticalDrag);
    trebleSlider.setColour(juce::Slider::thumbColourId, juce::Colours::darkgrey);
    trebleSlider.setColour(juce::Slider::rotarySliderFillColourId, juce::Colours::grey);
    trebleSlider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::white);
    trebleSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::darkgrey);
    trebleSlider.setTextBoxStyle(juce::Slider::TextEntryBoxPosition::TextBoxBelow, false, 50, 20);
    trebleSlider.setNumDecimalPlacesToDisplay(1);
    trebleAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            processorRef.params, "treble", trebleSlider);
    addAndMakeVisible(trebleSlider);


    trebleLabel.setText("Treble", juce::NotificationType::dontSendNotification);
    trebleLabel.setJustificationType(juce::Justification::centred);
    trebleLabel.setColour(juce::Label::ColourIds::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(trebleLabel);

    addAndMakeVisible(inMeterL);
    addAndMakeVisible(inMeterR);

    addAndMakeVisible(outMeterL);
    addAndMakeVisible(outMeterR);

    setSize (600, 400);
    startTimerHz(60);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::lightgrey);
    g.setFont(juce::Font(lnf.getCustomFont()));
}

void AudioPluginAudioProcessorEditor::timerCallback()
{
    inMeterL.setLevel(processorRef.getInRmsValue(0));
    inMeterR.setLevel(processorRef.getInRmsValue(1));
    inMeterL.repaint();
    inMeterR.repaint();

    outMeterL.setLevel(processorRef.getOutRmsValue(0));
    outMeterR.setLevel(processorRef.getOutRmsValue(1));
    outMeterL.repaint();
    outMeterR.repaint();
}


void AudioPluginAudioProcessorEditor::resized()
{
    inGainSlider.setBounds(85, 125, 70, 85);
    inGainLabel.setBounds(80, 105, 80, 15);

    outGainSlider.setBounds(445, 125, 70, 85);
    outGainLabel.setBounds(440, 105, 80, 15);

    bassSlider.setBounds(175, 125, 70, 85);
    bassLabel.setBounds(170, 105, 80, 15);

    middleSlider.setBounds(265, 125, 70, 85);
    middleLabel.setBounds(260, 105, 80, 15);

    trebleSlider.setBounds(355, 125, 70, 85);
    trebleLabel.setBounds(350, 105, 80, 15);

    ts1.setBounds(90, 260, 120, 40);
    ts3.setBounds(240, 260, 120, 40);
    rnn21.setBounds(390, 260, 120, 40);

    inMeterL.setBounds(30, 110, 5, 185);
    inMeterR.setBounds(37, 110, 5, 185);

    outMeterL.setBounds(563, 110, 5, 185);
    outMeterR.setBounds(570, 110, 5, 185);
}

void AudioPluginAudioProcessorEditor::sliderValueChanged(juce::Slider* slider)
{

}

void AudioPluginAudioProcessorEditor::buttonClicked(juce::Button* button)
{
    if (button == &rnn21)
        processorRef.selectModel("rnn21");
    else if (button == &ts1)
        processorRef.selectModel("ts1");
    else if (button == &ts3)
        processorRef.selectModel("ts3");
}