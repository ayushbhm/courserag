<template>
    
    


        <div
        class="levelcard">
           {{ levelcat }}
        </div>

     


 
</template>

<script setup>

defineProps({
    levelcat: String,
  });

</script>

<style scoped>

.levelcard {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 120px;
    width: 200px;
    border-radius: 16px;
    color: #1c0b07;
    font-family: 'Roboto', sans-serif;
    
    font-weight: 100;

    border: 3px solid transparent;
    background:
    linear-gradient(135deg, #f9f9fb, #f1f1f4) padding-box, /* Soft card background */
    linear-gradient(to bottom, rgba(225, 2, 191, 0.14), rgba(206, 203, 203, 0.1)) border-box;


    font-size: 1.5rem;
    font-weight: bold;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    
    cursor: pointer;
 
   
   
  }

  .levelcardinhomepage:hover {
    transform: scale(1.08);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    transition: transform .3s ease, box-shadow 0.2s ease;
  }
</style>