<template>
    <aside>
      <h2> Sidebar</h2>
  
      <div v-for="(week, index) in courseData" :key="index">
        <h3> {{ week.title }} </h3>
  
        <ul>
          <li
            v-for="(video, index) in week.videos"
            :key="video.title"
            @click="handleVideoClick(video.url)"
            style="cursor: pointer"
          >
          {{ index + 1}} {{ video.title }}
          </li>
        </ul>
      </div>
    </aside>
  </template>
  
  <script setup>
  defineProps({
    courseData: Array
  });
  
  // Emit the select-video event with the video URL when a video is clicked
  const emit = defineEmits(['select-video']);
  
  function handleVideoClick(videoUrl) {
    // Emit the select-video event when a video is clicked
    emit('select-video', videoUrl);
  }
  </script>
  