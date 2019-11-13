<template>
  <div>
    <!-- Template -->
    <div style="margin-top: -3.6rem">
      <strong style="margin-right: 10px">Base template:</strong>
      <el-radio-group :value="selection[domainOrSolver].template" fill="#F56C6C" size="small" style="margin-top: 5px; margin-bottom: 5px" @input="updateSelection({template: $event}, domainOrSolver)">
        <el-tooltip v-for="template in templates[domainOrSolver]" :key="template.name" effect="light" placement="bottom" :open-delay="500">
          <div class="tooltip" slot="content">
            <slot :name="template.name"></slot>
          </div>
          <el-radio-button :label="template.name"></el-radio-button>
        </el-tooltip>
      </el-radio-group>
    </div>

    <el-divider>Finetune characteristics (optional):</el-divider>

    <!-- Characteristics -->
    <div v-for="characteristic in characteristics[domainOrSolver]" :key="characteristic.name" style="margin-top: 10px">
      <span style="margin-right: 10px">{{characteristic.name}}:</span>
      <el-radio-group :value="selection[domainOrSolver].characteristics[characteristic.name]" size="small" style="margin-top: 5px; margin-bottom: 5px" @input="updateSelection({characteristics: {...selection[domainOrSolver].characteristics, [characteristic.name]: $event}}, domainOrSolver)">
        <el-tooltip v-for="(level, index) in characteristic.levels" :key="level" :disabled="level === '(none)'" effect="light" placement="top" :open-delay="500">
          <div class="tooltip" slot="content">
            <slot :name="level"></slot>
          </div>
          <el-radio-button :label="level" :disabled="index < templateLevelIndex[characteristic.name]"></el-radio-button>
        </el-tooltip>
      </el-radio-group>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    isSolver: {type: Boolean, default: false}
  },
  // data () {
  //   return {
  //     // templates: [
  //     //   {name: 'Domain', characteristics: {'Agent': 'MultiAgent', 'Concurrency': 'Parallel', 'Constraints': '(none)', 'Dynamics': 'Environment'}},
  //     //   {name: 'TestDomain', characteristics: {'Agent': 'SingleAgent', 'Concurrency': 'Sequential', 'Constraints': 'Constrained', 'Dynamics': 'UncertainTransitions'}}
  //     // ],
  //     // characteristics: [
  //     //   {name: 'Agent', levels: ['MultiAgent', 'SingleAgent']},
  //     //   {name: 'Concurrency', levels: ['Parallel', 'Sequential']},
  //     //   {name: 'Constraints', levels: ['(none)','Constrained']},
  //     //   {name: 'Dynamics', levels: ['Environment', 'Simulation', 'UncertainTransitions', 'EnumerableTransitions', 'DeterministicTransitions']},
  //     // ],
  //     // selection: {
  //     //   template: 'Domain',
  //     //   characteristics: {}
  //     // },
  //     // get selection() {
  //     //   console.log(localStorage.getItem('domainSelection') || {template: 'Domain', characteristics: {}})
  //     //   return JSON.parse(localStorage.getItem('domainSelection') || JSON.stringify({template: 'Domain', characteristics: {}}))
  //     // },
  //     // set selection (value) {
  //     //   console.log(value)
  //     //   localStorage.setItem('domainSelection', JSON.stringify(value))
  //     // }
  //   }
  // },
  beforeCreate () {
    this.$store.commit('loadSelection')
  },
  computed: {
    domainOrSolver () {
      return (this.isSolver ? 'solver' : 'domain')
    },
    templates () {
      return this.$store.state.templates
    },
    characteristics () {
      return this.$store.state.characteristics
    },
    selection () {
      return this.$store.state.selection
    },
    selectedTemplate () {
      return this.$store.getters.selectedTemplate
    },
    templateLevelIndex () {
      const levelIndex = {}
      this.characteristics[this.domainOrSolver].forEach(characteristic => {
        const index = characteristic.levels.indexOf(this.selectedTemplate[this.domainOrSolver].characteristics[characteristic.name])
        levelIndex[characteristic.name] = (index >= 0 ? index : 0)
      });
      return levelIndex
    }
  },
  methods: {
    updateSelection (selection, domainOrSolver) {
      this.$store.commit('updateSelection', {selection, domainOrSolver})
    }
  //   formatDoc (doc) {
  //     let format_doc = doc || ''

  //     format_doc.trim()
  //     format_doc = self.md.render(format_doc)
  //     //format_doc = format_doc.replace(/\n+/g, '<br/>')
  //     return format_doc

  //     format_doc = format_doc.replace(/\n*:::\s*(\w+)\n(.*?):::/gs, '<div class="$1 custom-block"><p>$2</p></div>')
  //     format_doc = format_doc.replace(/\n/g, '<br/>')
  //     return format_doc
  //   }
  },
  watch: {
    'selection.domain.template' (newVal) {
      this.updateSelection({characteristics: {...this.selectedTemplate['domain'].characteristics}}, 'domain')
    },
    'selection.solver.template' (newVal) {
      this.updateSelection({characteristics: {...this.selectedTemplate['solver'].characteristics}}, 'solver')
    }
    // 'selection.template': {
    //   immediate: false,
    //   handler (newVal) {
    //     this.updateSelection({characteristics: {...this.selectedTemplate.characteristics}})
    //     //this.selection.characteristics = Object.assign({}, this.selectedTemplate.characteristics)
    //     //this.$emit('update:selectedItemIndex', (this.autoSelectFirst ? 0 : -1))
    //   }
    // }
 }
}
</script>

<style scoped>
.tooltip {
  padding: 0px 20px;
  max-width: 600px;
  max-height: 40vh;
  overflow: auto
}
</style>