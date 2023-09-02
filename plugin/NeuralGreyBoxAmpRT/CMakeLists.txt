cmake_minimum_required(VERSION 3.22)

juce_add_plugin(${PROJECT_NAME}
    PRODUCT_NAME "NeuralGreyBoxAmpRT"
    COMPANY_NAME "Stepan Miklanek"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    PLUGIN_MANUFACTURER_CODE Stmk
    PLUGIN_CODE Ngba
    FORMATS VST3
)

target_sources(${PROJECT_NAME}
    PRIVATE
        PluginEditor.cpp
        PluginProcessor.cpp
        AmpModel.cpp
        ./nl_blocks/PreampBlock.cpp
        ./nl_blocks/PreampBlock.h
        ./nl_blocks/PowerampBlock.cpp
        ./nl_blocks/PowerampBlock.h
        ./nl_blocks/CondAmp.cpp
        ./nl_blocks/CondAmp.h
        ./ts_block/TonestackBlock.cpp
        ./ts_block/TonestackBlock.h
        ./ts_block/Component.cpp
        ./ts_block/Component.h
        ./ts_block/DKModel.cpp
        ./ts_block/DKModel.h
        ./gui/CustomLNF.h
        ./gui/Meter.h
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${LIB_DIR}/eigen
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        juce::juce_audio_utils
        nlohmann_json::nlohmann_json
        BinaryData
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

juce_add_binary_data(BinaryData
    SOURCES
        ./models/ts1.json
        ./models/ts3.json
        ./models/rnn21.json
        font/inter.ttf
)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

if (MSVC)
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        _SILENCE_CXX23_ALIGNED_STORAGE_DEPRECATION_WARNING)
endif()
