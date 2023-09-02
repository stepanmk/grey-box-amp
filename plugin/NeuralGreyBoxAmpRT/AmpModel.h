#pragma once

#include "./nl_blocks/PreampBlock.h"
#include "./nl_blocks/PowerampBlock.h"
#include "./ts_block/TonestackBlock.h"

namespace AmpModel {

    class AmpModel {
    public:
        AmpModel();

        void prepare(json &weights, float b, float m, float t);
        void processBlock(float *block, int numSamples);
        void setBass(float b);
        void setMiddle(float m);
        void setTreble(float t);

    private:

        Preamp preamp = Preamp(40);
        Tonestack tonestack = Tonestack(44100);
        Poweramp poweramp = Poweramp(8);

    };

}