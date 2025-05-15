<template>

<div class="container"> 

  <sidebar class="sidebar"
  :courseData="deepLearning" 
  @select-video="handleVideoSelect"
  
/>




<videoplayer class="videoplayer"

:url="selectedUrl"   

/>

</div>



</template>

<script setup>
import sidebar from '../components/sidebar.vue';
import videoplayer from '@/components/videoplayer.vue'
;
import deepLearning from "../assets/deepLearning.js";
import { ref, } from 'vue';


const selectedUrl = ref('');

function handleVideoSelect(url) {
console.log("Received video URL:", url);
selectedUrl.value = url;
console.log("Selected video URL:", selectedUrl.value);
}

</script>
<style scoped>

.container{
  display:flex;
  background: #f7f4f4 ;
  
}
.videoplayer{
  margin-top: 4%;
}
</style>
