#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
    : AudioProcessor (BusesProperties()
        #if ! JucePlugin_IsMidiEffect
        #if ! JucePlugin_IsSynth
            .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
        #endif
            .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
        #endif
    ),

    params(*this, nullptr, "params", {
        std::make_unique<juce::AudioParameterFloat>("ingain", "Input Gain", -12.f, 12.f, 0.f),
        std::make_unique<juce::AudioParameterFloat>("outgain", "Output Gain", -12.f, 12.f, 0.f),
        std::make_unique<juce::AudioParameterFloat>("bass", "Bass", 0.0f, 1.0f, 0.5f),
        std::make_unique<juce::AudioParameterFloat>("middle", "Middle", 0.0f, 1.0f, 0.5f),
        std::make_unique<juce::AudioParameterFloat>("treble", "Treble", 0.0f, 1.0f, 0.5f),
    })

{
    inGain = params.getRawParameterValue("ingain");
    outGain = params.getRawParameterValue("outgain");
    bassParam = params.getRawParameterValue("bass");
    middleParam = params.getRawParameterValue("middle");
    trebleParam = params.getRawParameterValue("treble");
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool AudioPluginAudioProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
    // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram()
{
    return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String AudioPluginAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

json AudioPluginAudioProcessor::getJsonContent(const char* data, int size)
{
    juce::MemoryInputStream jsonInputStream(data, size, false);
    json weights = json::parse(jsonInputStream.readEntireStreamAsString().toStdString());
    return weights;
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    rmsInL.reset(sampleRate, 0.1);
    rmsInR.reset(sampleRate, 0.1);
    rmsInL.setCurrentAndTargetValue(-100.f);
    rmsInR.setCurrentAndTargetValue(-100.f);

    rmsOutL.reset(sampleRate, 0.1);
    rmsOutR.reset(sampleRate, 0.1);
    rmsOutL.setCurrentAndTargetValue(-100.f);
    rmsOutR.setCurrentAndTargetValue(-100.f);

    json weights = getJsonContent(BinaryData::ts1_json, BinaryData::ts1_jsonSize);
    ts1[0].prepare(weights, bassParam->load(), middleParam->load(), trebleParam->load());
    ts1[1].prepare(weights, bassParam->load(), middleParam->load(), trebleParam->load());
    weights = getJsonContent(BinaryData::ts3_json, BinaryData::ts3_jsonSize);
    ts3[0].prepare(weights, bassParam->load(), middleParam->load(), trebleParam->load());
    ts3[1].prepare(weights, bassParam->load(), middleParam->load(), trebleParam->load());
    weights = getJsonContent(BinaryData::rnn21_json, BinaryData::rnn21_jsonSize);
    rnn21[0].loadWeights(weights, bassParam->load(), middleParam->load(), trebleParam->load());
    rnn21[1].loadWeights(weights, bassParam->load(), middleParam->load(), trebleParam->load());

}

void AudioPluginAudioProcessor::selectModel (const juce::String& model)
{
    if (model == "ts1") {
        ampNum = 0;
    }
    else if (model == "ts3") {
        ampNum = 1;
    }
    else if (model == "rnn21") {
        ampNum = 2;
    }
}

void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
#else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
#if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
#endif

    return true;
#endif
}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused (midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const auto numSamples = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();

    buffer.applyGain(juce::Decibels::decibelsToGain(inGain->load()));

    rmsInL.skip(numSamples);
    rmsInR.skip(numSamples);

    {
        const auto value = Decibels::gainToDecibels(buffer.getRMSLevel(0, 0, numSamples));
        if (value < rmsInL.getCurrentValue())
            rmsInL.setTargetValue(value);
        else
            rmsInL.setCurrentAndTargetValue(value);
    }

    {
        const auto value = Decibels::gainToDecibels(buffer.getRMSLevel(1, 0, numSamples));
        if (value < rmsInR.getCurrentValue())
            rmsInR.setTargetValue(value);
        else
            rmsInR.setCurrentAndTargetValue(value);
    }

    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* channelData = buffer.getWritePointer(ch);
        if (ampNum == 0) {
            ts1[ch].setBass(bassParam->load());
            ts1[ch].setMiddle(middleParam->load());
            ts1[ch].setTreble(trebleParam->load());
            ts1[ch].processBlock(channelData, numSamples);
        }
        else if (ampNum == 1) {
            ts3[ch].setBass(bassParam->load());
            ts3[ch].setMiddle(middleParam->load());
            ts3[ch].setTreble(trebleParam->load());
            ts3[ch].processBlock(channelData, numSamples);
        }
        else {
            rnn21[ch].setBass(bassParam->load());
            rnn21[ch].setMiddle(middleParam->load());
            rnn21[ch].setTreble(trebleParam->load());
            rnn21[ch].processBlock(channelData, numSamples);
        }
    }

    buffer.applyGain(juce::Decibels::decibelsToGain(outGain->load()));

    rmsOutL.skip(numSamples);
    rmsOutR.skip(numSamples);

    {
        const auto value = Decibels::gainToDecibels(buffer.getRMSLevel(0, 0, numSamples));
        if (value < rmsOutL.getCurrentValue())
            rmsOutL.setTargetValue(value);
        else
            rmsOutL.setCurrentAndTargetValue(value);
    }

    {
        const auto value = Decibels::gainToDecibels(buffer.getRMSLevel(1, 0, numSamples));
        if (value < rmsOutR.getCurrentValue())
            rmsOutR.setTargetValue(value);
        else
            rmsOutR.setCurrentAndTargetValue(value);
    }
}

float AudioPluginAudioProcessor::getInRmsValue(const int channel) const
{
    jassert(channel == 0 || channel == 1);
    if (channel == 0)
        return rmsInL.getCurrentValue();
    if (channel == 1)
        return rmsInR.getCurrentValue();
    return 0.f;
}

float AudioPluginAudioProcessor::getOutRmsValue(const int channel) const
{
    jassert(channel == 0 || channel == 1);
    if (channel == 0)
        return rmsOutL.getCurrentValue();
    if (channel == 1)
        return rmsOutR.getCurrentValue();
    return 0.f;
}

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    return new AudioPluginAudioProcessorEditor (*this);
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::ignoreUnused (destData);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ignoreUnused (data, sizeInBytes);
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}