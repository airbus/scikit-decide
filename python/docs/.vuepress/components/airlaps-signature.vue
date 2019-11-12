<template>
  <pre>
  <code class="language-python"><span class="token function">{{name}}</span>(
  <template v-for="p, i in sig.params">  {{p.name}}<template v-if="p.annotation">: {{adaptAnnotation(p.annotation)}}</template><template v-if="p.default"> <span class="token operator">=</span> <span class="token boolean">{{p.default}}</span></template><template v-if="i < sig.params.length - 1">,</template><br></template>)<template v-if="sig.return"> <span class="token operator">-></span> {{adaptAnnotation(sig.return)}}</template></code>
  </pre>
  <!-- <router-link to="/guide">world</router-link> -->
</template>

<script>
export default {
  props: {
    sig: {type: Object, default: () => ({params: []})},
    name: {type: String, default: ''}
//     selectedItemIndex: {type: Number, default: -1},
//     autoSelectFirst: {type: Boolean, default: false}
  },
  // data () {
  //   return {
  //     sig: {"params": [{"name": "test", "default": true, "annotation": "bool"}, {"name": "test2", "annotation": "str"}]}
  //   }
  // },
  computed: {
    selection () {
      return this.$store.state.selection
    },
    domainTypes () {
      return this.$store.getters.domainTypes
    }
  },
  methods: {
    adaptAnnotation (annotation) {
      if (this.selection['domain'].simplifySignatures) {
        let simplifiedAnnotation = annotation.replace(/\bD\.(\w+)\b/g, (match, type) => (this.domainTypes[type] !== undefined ? this.domainTypes[type].split('.').reverse()[0] : match))
        // Remove all unnecessary Union[...]
        const search = 'Union['
        let searchStart = 0
        while (true) {
          const start = simplifiedAnnotation.indexOf(search, searchStart)
          if (start < 0) {
            break
          }
          let bracketCounter = 0
          for(let i = start + search.length; i < simplifiedAnnotation.length; i++) {
            const char = simplifiedAnnotation.charAt(i)
            if (char === ',' && bracketCounter === 0) {
              searchStart = start + 1
              break
            } else if (char === '[') {
              bracketCounter++
            } else if (char === ']') {
              if (bracketCounter === 0) {
                simplifiedAnnotation = simplifiedAnnotation.slice(0, start) + simplifiedAnnotation.slice(start + search.length, i) + simplifiedAnnotation.slice(i + 1)
                break
              } else {
                bracketCounter--
              }
            }
          }
        }
        return simplifiedAnnotation
      }
      return annotation
    }
  }
}
</script>
