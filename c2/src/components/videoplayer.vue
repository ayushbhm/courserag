<script setup>
const props = defineProps({
  url: String
})

function getEmbedUrl(url) {
  if (!url) return '';
  
  // Check if it's a normal YouTube URL
  if (url.includes('watch?v=')) {
    return url.replace('watch?v=', 'embed/');
  }

  // Optional: if the URL is already an embed link or something else, return it as is
  return url;
}
</script>

<template>
  <div>
    <iframe 
      v-if="props.url"
      width="560" 
      height="315" 
      :src="getEmbedUrl(props.url)" 
      frameborder="0" 
      allowfullscreen>
    </iframe>
  </div>
</template>
