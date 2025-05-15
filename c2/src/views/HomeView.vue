<template>

<div class="card-container">

<router-link v-for="category in levelcat"  :to="`/course`" style="text-decoration: none; color: inherit;"  >

  <levelcard   :levelcat="category.name" class="levelcardinhomepage" />

</router-link>


 
  

</div>

</template>

<script setup>

import levelcard from '@/components/levelcard.vue';

import {ref} from 'vue'

const levelmode=ref('')

const levelcat = [
    {
        id:1, name: 'Foundation'
    },
    {
        id:2,
        name:'Diploma'

    },
    {
        id:3,
        name:'Degree'
    }
]

function selectCat(category){

}




</script>
<style scoped>
.background{
  min-height: 100vh;
  background: linear-gradient(215deg, #f5f7fa, #e3e3e3);
  display: flex;
  flex-direction: row;
}

.blob {
  position: absolute;
  top: -100px;
  left: -100px;
  width: 300px;
  height: 300px;
  background: radial-gradient(circle, rgba(255,0,150,0.3) 0%, rgba(0,255,255,0.3) 100%);
  filter: blur(80px);
  z-index: 0;
}

.card-container{
display: flex;
flex-direction: row;
gap:5%;
justify-content: center;

}
</style>
