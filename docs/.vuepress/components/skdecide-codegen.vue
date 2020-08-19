<template>
  <div style="margin-top: 30px">
  
  <el-row type="flex" justify="space-between" align="middle">
    <el-switch v-model="isSolver" inactive-text="Create Domain" active-text="Create Solver" inactive-color="#409EFF" active-color="#409EFF"></el-switch>

    <el-button round icon="el-icon-copy-document" size="small" @click="copyCode">
      <strong>Copy code</strong>
    </el-button>
  </el-row>

  <slot v-if="isSolver" name="SolverSummary"></slot>

  <pre>
  <code id="gencode" class="language-python"><template v-if="!isSolver"><span class="token keyword">from</span> enum <span class="token keyword">import</span> Enum</template>
  <span class="token keyword">from</span> typing <span class="token keyword">import</span> <span class="token operator">*</span>
  
  <span class="token keyword">from</span> skdecide <span class="token keyword">import</span> <span class="token operator">*</span>
  <span class="token keyword">from</span> skdecide.builders.domain <span class="token keyword">import</span> <span class="token operator">*</span><template v-if="isSolver">
  <span class="token keyword">from</span> skdecide.builders.solver <span class="token keyword">import</span> <span class="token operator">*</span></template>

  <template v-if="!isSolver">
  <span class="token comment"># Example of State type (adapt to your needs)</span>
  <span class="token keyword">class</span> <span class="token class-name">State</span>(NamedTuple):
      x: <span class="token builtin">int</span>
      y: <span class="token builtin">int</span>
  

  <span class="token comment"># Example of Action type (adapt to your needs)</span>
  <span class="token keyword">class</span> <span class="token class-name">Action</span>(Enum):
      up <span class="token operator">=</span> <span class="token number">0</span>
      down <span class="token operator">=</span> <span class="token number">1</span>
      left <span class="token operator">=</span> <span class="token number">2</span>
      right <span class="token operator">=</span> <span class="token number">3</span>
  
  </template>
  <span class="token keyword">class</span> <span class="token class-name">D</span>({{ domainInheritance }}):
      <template v-if="isSolver">pass</template><template v-else>T_state <span class="token operator">=</span> State  <span class="token comment"># Type of states</span>
      T_observation <span class="token operator">=</span> T_state  <span class="token comment"># Type of observations</span>
      T_event <span class="token operator">=</span> Action  <span class="token comment"># Type of events</span>
      T_value <span class="token operator">=</span> <span class="token builtin">float</span>  <span class="token comment"># Type of transition values (rewards or costs)</span>
      T_info <span class="token operator">=</span> <span class="token boolean">None</span>  <span class="token comment"># Type of additional information in environment outcome</span></template>
  
  
  <span class="token keyword">class</span> <template v-if="isSolver"><span class="token class-name">MySolver</span>({{ solverInheritance }}):
      T_domain <span class="token operator">=</span> D</template><template v-else><span class="token class-name">MyDomain</span>(D):</template>
      <template v-for="[characteristic, level] in Object.entries({...{'default': domainOrSolver}, ...selection[domainOrSolver].characteristics}).filter(([k, v]) => v != '(none)')"><template v-for="method in methods[domainOrSolver][level]">
      <span class="token keyword">def</span> <span class="token function">{{method}}</span>(<template v-for="p, i in signatures[method].params">{{p.name}}<template v-if="p.annotation">: {{adaptAnnotation(p.annotation)}}</template><span v-show="p.default"> <span class="token operator">=</span> <span class="token boolean">{{p.default}}</span></span><span v-show="i < signatures[method].params.length - 1">, </span></template>)<template v-if="signatures[method].return"> <span class="token operator">-></span> {{adaptAnnotation(signatures[method].return)}}</template>:
          <span class="token keyword">pass</span>
      </template></template>
  </code>
  </pre>
  </div>
</template>

<script>
export default {
  // props: {
  //   sig: {type: Object, default: () => ({params: []})},
  //   name: {type: String, default: ''}
//     selectedItemIndex: {type: Number, default: -1},
//     autoSelectFirst: {type: Boolean, default: false}
  // },
  data () {
    return {
      isSolver: false
    }
  },
  computed: {
    domainOrSolver () {
      return (this.isSolver ? 'solver' : 'domain')
    },
    selection () {
      return this.$store.state.selection
    },
    selectedTemplate () {
      return this.$store.getters.selectedTemplate
    },
    domainInheritance () {
      const inheritance = [this.selection['domain'].template, ...Object.entries(this.selection['domain'].characteristics).filter(c => this.isFinetuned(c[0], c[1], 'domain')).map(c => c[1])]
      return inheritance.join(', ')
    },
    solverInheritance () {
      const inheritance = [this.selection['solver'].template, ...Object.entries(this.selection['solver'].characteristics).filter(c => this.isFinetuned(c[0], c[1], 'solver')).map(c => c[1])]
      return inheritance.join(', ')
    },
    methods () {
      return this.$store.state.methods
    },
    domainTypes () {
      return this.$store.getters.domainTypes
    },
    signatures () {
      return this.$store.state.signatures[this.domainOrSolver]
    }
  },
  methods: {
    isFinetuned (characteristic, level, domainOrSolver) {
      return this.selectedTemplate[domainOrSolver].characteristics[characteristic] !== level
    },
    adaptAnnotation (annotation) {
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
      return adaptedAnnotation
    },
    copyCode () {
      const copyText = document.getElementById('gencode').textContent
      const textArea = document.createElement('textarea')
      textArea.textContent = copyText
      document.body.append(textArea)
      textArea.select()
      document.execCommand('copy')
    }
  }
}
</script>

<style>
textarea {
  position: absolute;
  left: -100%;
}
</style>