<script setup>
import { Icon } from '@iconify/vue';
import { ref, onMounted } from 'vue'
import ModelNames from './components/ModelNames.vue';


// let sound = document.createElement('audio');
// sound.id = 'audio-player';
// sound.controls = 'controls';
// sound.src = './assets/audio/B1_M3_T7/1_target_LSTM40_pwr_GRU8_B1_M3_T7_G5_file_9.wav';

onMounted(() => {
    addAllSounds();
})


function addAllSounds()
{    
    let selectedFiles = [4, 8, 3, 7, 9, 1, 2, 5];
    addSetting(selectedFiles, 'B1_M3_T7', 'setting-1');
    addSetting(selectedFiles, 'B10_M0_T0', 'setting-2');
    addSetting(selectedFiles, 'B3_M7_T1', 'setting-3');
    addSetting(selectedFiles, 'B0_M0_T10', 'setting-4');

}

function addSetting(selectedFiles, settingName, elementName)
{
    const models = [
        '1_target_LSTM48_rnn_only',
        '3_targets_LSTM48_rnn_only',
        '21_targets_LSTM48_rnn_only_fin',
        '1_target_LSTM40_pwr_GRU8',
        '3_targets_LSTM40_pwr_GRU8',
    ]
    let setting = document.getElementById(elementName);
    selectedFiles.forEach((file, idx) => {
        let sampleNumber = document.createElement('div');
        sampleNumber.innerText = `Sample ${idx + 1}`;
        sampleNumber.style.display = 'flex';
        sampleNumber.style.justifyContent = 'center';
        sampleNumber.style.alignItems = 'center';
        setting.appendChild(sampleNumber);
        
        let inputDiv = document.createElement('div');
        inputDiv.style.display = 'flex';
        inputDiv.style.justifyContent = 'center';
        inputDiv.style.alignItems = 'center';
        setting.appendChild(inputDiv);

        let inputSound = document.createElement('audio');
        inputSound.src = `./src/assets/audio/${settingName}/${settingName}_G5_file_${file}_input.wav`
        inputSound.controls = 'controls';
        inputSound.style.width = '7rem';
        inputSound.style.height = '2rem';
        inputDiv.appendChild(inputSound);
        
        let refDiv = document.createElement('div');
        refDiv.style.display = 'flex';
        refDiv.style.justifyContent = 'center';
        refDiv.style.alignItems = 'center';
        setting.appendChild(refDiv);

        let refSound = document.createElement('audio');
        refSound.src = `./src/assets/audio/${settingName}/${settingName}_G5_file_${file}.wav`
        refSound.controls = 'controls';
        refSound.style.width = '7rem';
        refSound.style.height = '2rem';
        refDiv.appendChild(refSound);

        models.forEach((modelName) => {
            let modelDiv = document.createElement('div');
            modelDiv.style.display = 'flex';
            modelDiv.style.justifyContent = 'center';
            modelDiv.style.alignItems = 'center';
            setting.appendChild(modelDiv);

            let modelSound = document.createElement('audio');
            modelSound.src = `./src/assets/audio/${settingName}/${modelName}_${settingName}_G5_file_${file}.wav`
            modelSound.controls = 'controls';
            modelSound.style.width = '7rem';
            modelSound.style.height = '2rem';
            modelDiv.appendChild(modelSound);
        })
    })
}


</script>

<template>

<div id="main-container" class="min-h-screen flex justify-center">

    <div id="content" class="flex flex-col items-start justify-start p-8 gap-2 px-[22rem] max-w-[120rem]">

        <p class="text-4xl font-semibold">Neural Grey-Box Guitar Amplifier Modelling with Limited Data</p>

        <p class="w-full">Štěpán Miklánek, Alec Wright, Vesa Välimäki and Jiří Schimmel</p>

        <p class="w-full font-semibold mt-4 text-lg">Abstract</p>
        
        <p class="text-justify">This paper combines recurrent neural networks 
        (RNNs) with the discretised Kirchhoff nodal analysis (DK-method) to create a
        grey-box guitar amplifier model. Both the objective and subjective
        results suggest that the proposed model is able to outperform
        a baseline black-box RNN model in the task of modelling a guitar
        amplifier, including realistically recreating the behaviour of the
        amplifier equaliser circuit, whilst requiring significantly less training
        data. Furthermore, we adapt the linear part of the DK-method
        in a deep learning scenario to derive multiple state-space filters simultaneously.
        We frequency sample the filter transfer functions in
        parallel and perform frequency domain filtering to considerably reduce
        the required training times compared to recursive state-space
        filtering. This study shows that it is a powerful idea to separately
        model the linear and nonlinear parts of a guitar amplifier using
        supervised learning.</p>

        <p class="w-full font-semibold mt-4 text-lg">Resources</p>
        <div class="w-full flex flex-col xl:flex-row gap-4">
            <a href="" class="pl-2 pr-4 h-9 flex flex-row items-center justify-center gap-1 bg-neutral-600 rounded-lg hover:bg-neutral-900 text-white">
                <Icon icon="ph:file-text" width="26"/>
                <p class="">Paper</p>
            </a>

            <a href="https://github.com/stepanmk/grey-box-amp/tree/master/python" class="pl-2 pr-4 h-9 flex flex-row items-center justify-center gap-1 bg-neutral-600 rounded-lg hover:bg-neutral-900 text-white">
                <Icon icon="ph:github-logo" width="26"/>
                <p class="">Code</p>
            </a>
            
            <a href="https://zenodo.org/record/7970723" class="pl-2 pr-4 h-9 flex flex-row items-center justify-center gap-1 bg-neutral-600 rounded-lg hover:bg-neutral-900 text-white">
                <Icon icon="ph:file-audio" width="26"/>
                <p class="">Dataset</p>
            </a>
        </div>

        <p class="w-full font-semibold mt-4 text-lg">Listening examples (unseen tone stack settings)</p>
        
        <ModelNames title="Bass: 1, Middle: 3, Treble: 7 (MUSHRA setting 1)"></ModelNames>
        <div id="setting-1" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>
        
        <ModelNames title="Bass: 10, Middle: 0, Treble: 0 (MUSHRA setting 2)"></ModelNames>
        <div id="setting-2" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>
        
        <ModelNames title="Bass: 3, Middle: 7, Treble: 1"></ModelNames>
        <div id="setting-3" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>  

        <ModelNames title="Bass: 0, Middle: 0, Treble: 10"></ModelNames>
        <div id="setting-4" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>  
    
    </div>
</div>



</template>

<style scoped>

html, body {
    overflow-x: hidden;
}

</style>
