#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "./nl_blocks/CondAmp.h"
#include "AmpModel.h"

#include "BinaryData.h"


//==============================================================================
class AudioPluginAudioProcessor  : public juce::AudioProcessor
{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    juce::AudioProcessorValueTreeState params;

    void selectModel (const juce::String& model);
    float getInRmsValue(int channel) const;
    float getOutRmsValue(int channel) const;

private:
    //==============================================================================
    json getJsonContent(const char* data, int size);

    std::atomic<float>* inGain = nullptr;
    std::atomic<float>* outGain = nullptr;
    std::atomic<float>* bassParam = nullptr;
    std::atomic<float>* middleParam = nullptr;
    std::atomic<float>* trebleParam = nullptr;

    AmpModel::AmpModel ts1[2] = {AmpModel::AmpModel(), AmpModel::AmpModel()};
    AmpModel::AmpModel ts3[2] = {AmpModel::AmpModel(), AmpModel::AmpModel()};
    AmpModel::CondAmp rnn21[2] = {AmpModel::CondAmp(48), AmpModel::CondAmp(48)};

    juce::LinearSmoothedValue<float> rmsInL, rmsInR;
    juce::LinearSmoothedValue<float> rmsOutL, rmsOutR;

    int ampNum = 0;
    bool firstRun = true;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};