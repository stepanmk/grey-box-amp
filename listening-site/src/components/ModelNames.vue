<script setup>
const props = defineProps({
    title: String,
})


</script>

<template>
    <p class="w-full text-base flex items-center justify-center mt-4 font-semibold border-b">{{ title }}</p>
    <div class="w-full flex flex-row justify-around text-white">
        <p class="w-[7rem]"></p>
        <p class="flex items-center justify-center rounded-md w-[7rem] bg-neutral-600">Input</p>
        <p class="flex items-center justify-center rounded-md w-[7rem] bg-green-600">Reference</p>
        <p class="flex items-center justify-center rounded-md w-[7rem] bg-cyan-600">RNN1</p>
        <p class="flex items-center justify-center rounded-md w-[7rem] bg-cyan-600">RNN3</p>
        <p class="flex items-center justify-center rounded-md w-[7rem] bg-cyan-600">RNN21</p>
        <p class="flex items-center justify-center rounded-md w-[7rem] bg-violet-600">TS1</p>
        <p class="flex items-center justify-center rounded-md w-[7rem] bg-violet-600">TS3</p>
    </div>
</template>