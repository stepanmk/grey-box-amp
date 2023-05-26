<script setup>
import { Icon } from '@iconify/vue';

</script>

<template>

<div id="main-container" class="min-h-screen flex flex-col items-start justify-start py-16 gap-2 px-[5rem] xl:px-[20rem]">

    <p class="text-4xl font-semibold">Neural Grey-Box Guitar Amplifier Modelling with Limited Data</p>

    <p class="w-full">Štěpán Miklánek, Alec Wright, Vesa Välimäki and Jiří Schimmel</p>

    <p class="w-full font-semibold mt-4">Abstract</p>
    
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

    <p class="w-full font-semibold mt-4">Resources</p>
    <div class="w-full flex flex-row gap-4">
        <button class="pl-2 pr-4 h-10 flex flex-row items-center justify-center gap-1 bg-neutral-600 rounded-lg hover:bg-neutral-900 text-white">
            <Icon icon="ph:file-text" width="32"/>
            <p class="font-semibold">Paper</p>
        </button>

        <button class="pl-2 pr-4 h-10 flex flex-row items-center justify-center gap-1 bg-neutral-600 rounded-lg hover:bg-neutral-900 text-white">
            <Icon icon="ph:github-logo" width="32"/>
            <p class="font-semibold">Code</p>
        </button>
        
        <button class="pl-2 pr-4 h-10 flex flex-row items-center justify-center gap-1 bg-neutral-600 rounded-lg hover:bg-neutral-900 text-white">
            <Icon icon="ph:file-audio" width="32"/>
            <p class="font-semibold">Dataset</p>
        </button>
    </div>

    <p class="w-full font-semibold mt-4">Listening examples</p>

</div>



</template>

<style scoped>

html, body {
    overflow-x: hidden;
}

</style>
