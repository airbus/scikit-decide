<template>
  <pre>
  <code class="language-python">{{name}}(
  <template v-for="p, i in sig.params">  {{p.name}}<span v-if="p.annotation">: {{p.annotation}}</span><span v-if="p.default"> = {{p.default}}</span><span v-if="i < sig.params.length - 1">,</span><br></template>)<span v-if="sig.return"> -> {{sig.return}}</span></code>
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
    formatted_sig () {
      const formatted_params = this.sig.params.map(p => '    ' + p.name + (p.annotation ? ': ' + p.annotation : '') + (p.default ? ' = ' + p.default : '')) .join(',\n')
      const return_annotation = (this.sig.return ? ' -> ' + this.sig.return : '')
      return this.name + (formatted_params.length > 0 ? '(\n' + formatted_params + '\n  )' : '()') + return_annotation
    }
  },
//   methods: {
//     showAlert (msg) {
//       alert(msg)
//     }
//   },
//   watch: {
//     results (val) {
//       this.$emit('update:selectedItemIndex', (this.autoSelectFirst ? 0 : -1))
//     }
//  }
}
</script>
