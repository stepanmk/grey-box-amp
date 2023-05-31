<script setup>
import { Icon } from '@iconify/vue';
import { ref, onMounted } from 'vue'
import ModelNames from './components/ModelNames.vue';

onMounted(() => {
    addAllSounds();
})

function addAllSounds()
{   
    let models = [
        '1_target_LSTM48_rnn_only',
        '3_targets_LSTM48_rnn_only',
        '21_targets_LSTM48_rnn_only_fin',
        '1_target_LSTM40_pwr_GRU8',
        '3_targets_LSTM40_pwr_GRU8',
    ];
    let selectedFiles = [4, 8, 3, 1, 2, 5];
    addSetting(selectedFiles, 'B1_M3_T7', 'setting-1', models);
    addSetting(selectedFiles, 'B10_M0_T0', 'setting-2', models);

    selectedFiles = [7, 9];
    addSetting(selectedFiles, 'B3_M7_T1', 'setting-3', models);
    addSetting(selectedFiles, 'B0_M0_T10', 'setting-4', models);
    addSetting(selectedFiles, 'B10_M0_T10', 'setting-5', models);

    let settings = ['B0_M5_T5', 'B2_M5_T5', 'B4_M5_T5', 'B6_M5_T5', 'B8_M5_T5', 'B10_M5_T5'];
    addEval(settings, 'setting-6', '');
    addEval(settings, 'setting-7', '1_target_LSTM40_pwr_GRU8');
    addEval(settings, 'setting-8', '3_targets_LSTM40_pwr_GRU8');

    settings = ['B5_M0_T5', 'B5_M2_T5', 'B5_M4_T5', 'B5_M6_T5', 'B5_M8_T5', 'B5_M10_T5'];
    addEval(settings, 'setting-9', '');
    addEval(settings, 'setting-10', '1_target_LSTM40_pwr_GRU8');
    addEval(settings, 'setting-11', '3_targets_LSTM40_pwr_GRU8');

    settings = ['B5_M5_T0', 'B5_M5_T2', 'B5_M5_T4', 'B5_M5_T6', 'B5_M5_T8', 'B5_M5_T10'];
    addEval(settings, 'setting-12', '');
    addEval(settings, 'setting-13', '1_target_LSTM40_pwr_GRU8');
    addEval(settings, 'setting-14', '3_targets_LSTM40_pwr_GRU8');
}

function addEval(settings, elementName, modelName)
{
    let setting = document.getElementById(elementName);

    settings.forEach((settingName) => {
        let modelDiv = document.createElement('div');
        modelDiv.style.display = 'flex';
        modelDiv.style.justifyContent = 'center';
        modelDiv.style.alignItems = 'center';
        setting.appendChild(modelDiv);

        let modelSound = document.createElement('audio');
        if (modelName === '')
        {
            modelSound.src = `./audio/add/${settingName}/${settingName}_G5_file_9.wav`;
        }
        else
        {
            modelSound.src = `./audio/add/${settingName}/${modelName}_${settingName}_G5_file_9.wav`;
        }
        modelSound.controls = 'controls';
        modelSound.style.width = '7rem';
        modelSound.style.height = '2rem';
        modelDiv.appendChild(modelSound);

        setting.appendChild(modelDiv);
    })
}

function addSetting(selectedFiles, settingName, elementName, models)
{
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
        inputSound.src = `./audio/${settingName}/${settingName}_G5_file_${file}_input.wav`
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
        refSound.src = `./audio/${settingName}/${settingName}_G5_file_${file}.wav`
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
        modelSound.src = `./audio/${settingName}/${modelName}_${settingName}_G5_file_${file}.wav`
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

    <div id="content" class="flex flex-col items-start justify-start p-8 gap-2 max-w-[75rem]">

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

        <p class="text-justify">The baseline RNN model is an LSTM of hidden size 48. The TS model (proposed) 
            is composed of an LSTM of hidden size 40, followed by a white-box tone stack model and a GRU of hidden size 8.
            The numbering denotes the number of tone stack settings the models have seen during training.
            A single tone stack setting corresponds to 4 minutes of audio used during training.
            The dataset was recorded using the speaker output of the Marshall JVM 410H guitar amplifier connected to a reactive load.
            A speaker cabinet impulse response was not applied to the presented sounds.
        </p>
        
        <ModelNames title="Bass: 1, Middle: 3, Treble: 7 (MUSHRA setting 1)"></ModelNames>
        <div id="setting-1" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>
        
        <ModelNames title="Bass: 10, Middle: 0, Treble: 0 (MUSHRA setting 2)"></ModelNames>
        <div id="setting-2" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>
        
        <ModelNames title="Bass: 3, Middle: 7, Treble: 1"></ModelNames>
        <div id="setting-3" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>  

        <ModelNames title="Bass: 0, Middle: 0, Treble: 10"></ModelNames>
        <div id="setting-4" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>  

        <ModelNames title="Bass: 10, Middle: 0, Treble: 10"></ModelNames>
        <div id="setting-5" class="w-full grid grid-cols-8 gap-y-2 border-b pb-2"></div>
        
        <p class="w-full font-semibold mt-4 text-lg">Listening examples (further evaluation of the proposed model)</p>
        <p class="text-justify">The following examples show that the proposed model generalises well to additional unseen tone stack permutations. 
            The TS1 model was trained only on a single tone stack setting (Bass: 5, Middle: 5, Treble: 5), whereas the TS3 model was trained on three
            tone stack permutations (Bass: 0, Middle: 0, Treble: 0), (Bass: 5, Middle: 5, Treble: 5), and (Bass: 10, Middle: 10, Treble: 10).
            This corresponds to 4 minutes of training data for the TS1 model and 12 minutes of training data for the TS3 model.
        </p>

        <!-- bass -->
        <p class="w-full text-base flex items-center justify-center mt-4 font-semibold border-b">Bass control varied from 0 to 10, with Middle and Treble controls set to 5 in all cases.</p>
        <div class="w-full flex flex-row justify-around">
            <p class="w-[7rem]"></p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Bass: 0</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Bass: 2</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Bass: 4</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Bass: 6</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Bass: 8</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Bass: 10</p>
        </div>

        <div id="setting-6" class="w-full flex flex-row justify-around items-center">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-green-600 text-white">Reference</p>
        </div>

        <div id="setting-7" class="w-full flex flex-row justify-around items-center">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-violet-600 text-white">TS1</p>
        </div>
        
        <div id="setting-8" class="w-full flex flex-row justify-around items-center border-b pb-2">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-violet-600 text-white">TS3</p>
        </div>

        <!-- mid -->
        <p class="w-full text-base flex items-center justify-center mt-4 font-semibold border-b">Middle control varied from 0 to 10, with Bass and Treble controls set to 5 in all cases.</p>
        <div class="w-full flex flex-row justify-around">
            <p class="w-[7rem]"></p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Middle: 0</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Middle: 2</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Middle: 4</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Middle: 6</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Middle: 8</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Middle: 10</p>
        </div>

        <div id="setting-9" class="w-full flex flex-row justify-around items-center">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-green-600 text-white">Reference</p>
        </div>

        <div id="setting-10" class="w-full flex flex-row justify-around items-center">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-violet-600 text-white">TS1</p>
        </div>
        
        <div id="setting-11" class="w-full flex flex-row justify-around border-b pb-2 items-center">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-violet-600 text-white">TS3</p>
        </div>

        <!-- mid -->
        <p class="w-full text-base flex items-center justify-center mt-4 font-semibold border-b">Treble control varied from 0 to 10, with Bass and Middle controls set to 5 in all cases.</p>
        <div class="w-full flex flex-row justify-around">
            <p class="w-[7rem]"></p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Treble: 0</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Treble: 2</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Treble: 4</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Treble: 6</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Treble: 8</p>
            <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-200">Treble: 10</p>
        </div>

        <div id="setting-12" class="w-full flex flex-row justify-around items-center">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-green-600 text-white">Reference</p>
        </div>

        <div id="setting-13" class="w-full flex flex-row justify-around items-center">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-violet-600 text-white">TS1</p>
        </div>
        
        <div id="setting-14" class="w-full flex flex-row justify-around border-b pb-2 items-center">
            <p class="flex items-center justify-center rounded-md w-[7rem] h-6 bg-violet-600 text-white">TS3</p>
        </div>

    </div>
</div>



</template>

<style scoped>

html, body {
    overflow-x: hidden;
}

</style>
