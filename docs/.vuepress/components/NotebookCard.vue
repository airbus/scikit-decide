<template>
  <a v-bind:href="reference">
    <div class="md-card" :class="{ 'show-border': !hideBorder }">
      <div v-if="image" class="card-image">
        <img :src="$withBase(image)" alt="" />
      </div>
      <div class="card-content">
        <slot></slot>
      </div>
    </div>
  </a>
</template>

<script>
export default {
  props: ["image", "hideBorder", "reference"],
};
</script>

<style lang="stylus">
.md-card {
  background-color: #c8f4ff;
  margin-top: 1em;
  min-height: 150px;
  display: flex;
  flex-direction: row;
  align-items: stretch;

  .card-image {
    display: flex;
    align-items: center;
    padding: 0.5rem;

    img {
      max-width: 150px;
      max-height: 150px;
      height: 150px;
      border: 1px solid #eee;
      border-radius: 0.2rem;
      object-fit: cover;
    }
  }

  .card-content {
    padding: 0.5rem;
    flex-grow: 1;

    p {
      line-height: normal;
      -webkit-margin-before: 0em;
      -webkit-margin-after: 0.5em;
    }

    blockquote {
      font-size: 1rem;
    }
  }
}

.md-card.show-border {
  border: 1px solid #eee;
  border-radius: 0.5rem;
  box-shadow: 0 5px 15px -5px rgba(0, 0, 0, 0.1);
  padding: 0.5rem;
}

@media (max-width: 419px) {
  .md-card {
    display: flex;
    flex-direction: column;
    margin: 1rem -0.5rem;
    padding: 0.5rem;
    box-sizing: border-box;

    .card-image {
      flex-grow: 1;
      align-items: stretch;

      img {
        width: 100%;
        max-width: 400px;
      }
    }

    .card-content {
      pre {
        margin: 0;
        border-radius: 6px;
      }
    }
  }
}
</style>
