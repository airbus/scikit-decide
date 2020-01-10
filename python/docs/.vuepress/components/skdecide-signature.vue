<template>
  <pre>
  <code class="language-python"><span class="token function">{{name}}</span>(
  <template v-for="p, i in sig.params">  {{p.name}}<template v-if="p.annotation">: <span v-html="adaptAnnotationHtml(p.annotation)"></span></template><template v-if="p.default"> <span class="token operator">=</span> <span class="token boolean">{{p.default}}</span></template><template v-if="i < sig.params.length - 1">,</template><br></template>)<template v-if="sig.return"> <span class="token operator">-></span> <span v-html="adaptAnnotationHtml(sig.return)"></span></template></code>
  </pre>
</template>

<script>
export default {
  props: {
    sig: {type: Object, default: () => ({params: []})},
    name: {type: String, default: ''}
  },
  data () {
    return {
      needsLinksUpdate: false
    }
  },
  computed: {
    selection () {
      return this.$store.state.selection
    },
    objects () {
      return this.$store.state.objects
    },
    domainTypes () {
      return this.$store.getters.domainTypes
    },
  },
  methods: {
    adaptAnnotationHtml (annotation) {
      let adaptedAnnotation = annotation
      if (this.selection['domain'].simplifySignatures) {
        adaptedAnnotation = adaptedAnnotation.replace(/\bD\.(\w+)\b/g, (match, type) => (this.domainTypes[type] !== undefined ? this.domainTypes[type].split('.').reverse()[0] : match))
        // Remove all unnecessary Union[...]
        const search = 'Union['
        let searchStart = 0
        while (true) {
          const start = adaptedAnnotation.indexOf(search, searchStart)
          if (start < 0) {
            break
          }
          let bracketCounter = 0
          for(let i = start + search.length; i < adaptedAnnotation.length; i++) {
            const char = adaptedAnnotation.charAt(i)
            if (char === ',' && bracketCounter === 0) {
              searchStart = start + 1
              break
            } else if (char === '[') {
              bracketCounter++
            } else if (char === ']') {
              if (bracketCounter === 0) {
                adaptedAnnotation = adaptedAnnotation.slice(0, start) + adaptedAnnotation.slice(start + search.length, i) + adaptedAnnotation.slice(i + 1)
                break
              } else {
                bracketCounter--
              }
            }
          }
        }
      }
      //adaptedAnnotation = adaptedAnnotation.replace(/\w+/g, match => (this.objects[match] !== undefined ? ('<router-link to="' + this.objects[match] + '">' + match + '</router-link>') : match))
      //adaptedAnnotation = adaptedAnnotation.replace(/\w+/g, match => (this.objects[match] !== undefined ? ('<a href="' + this.objects[match] + '">' + match + '</a>') : match))
      adaptedAnnotation = adaptedAnnotation.replace(/\w+/g, match => (this.objects[match] !== undefined ? ('<a class="linkto" data-link="' + this.objects[match] + '">' + match + '</a>') : match))
      this.needsLinksUpdate = true
      this.$nextTick(this.updateLinks)
      return adaptedAnnotation
    },
    updateLinks () {
      if (this.needsLinksUpdate) {
        const links = [...document.getElementsByClassName("linkto")]
        links.forEach(link => link.addEventListener("click", () => this.$router.push(link.getAttribute("data-link"))))
        this.needsLinksUpdate = false
      }
    }
  }
}
</script>

<style>

.linkto {
  cursor: pointer;
}

</style>