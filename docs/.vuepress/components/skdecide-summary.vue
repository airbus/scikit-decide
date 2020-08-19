<template>
  <div style="margin: 10px 0px">
    <span>{{ isSolver ? 'Solver' : 'Domain' }} specification:</span>
    
    <div style="margin-top: 5px; margin-bottom: 10px">
    <!-- Template -->
    <el-tag type="danger" effect="dark" style="margin-bottom: 5px">
      <strong>{{ selection[domainOrSolver].template }}</strong>
    </el-tag>
    
    <!-- Characteristics -->
    <el-tag v-for="[characteristic, level] in Object.entries(selection[domainOrSolver].characteristics).filter(([k, v]) => !selection[domainOrSolver].showFinetunedOnly || isFinetuned(k, v, domainOrSolver))" :key="characteristic" :effect="isFinetuned(characteristic, level, domainOrSolver) ? 'dark' : 'plain'" class="tag">
      <el-tag size="mini" class="subtag">{{ characteristic }}</el-tag>
      <strong>{{ level }}</strong>
    </el-tag>
    </div>

    <el-button type="info" round icon="el-icon-edit" size="small" style="margin-right: 15px; margin-bottom: 5px" @click="specDialogVisible = true">
      <strong>Edit</strong>
    </el-button>

    <el-checkbox :value="selection[domainOrSolver].showFinetunedOnly" @input="updateSelection({showFinetunedOnly: $event}, domainOrSolver)">Only show finetuned characteristics</el-checkbox>

    <el-checkbox v-if="!isSolver" :value="selection[domainOrSolver].simplifySignatures" @input="updateSelection({simplifySignatures: $event}, domainOrSolver)">Simplify signatures</el-checkbox>

    <!-- Spec dialog -->
    <el-dialog :title="'Edit ' + domainOrSolver + ' specification'" :visible.sync="specDialogVisible">
      <iframe v-if="specDialogVisible" :src="$withBase('/guide/_' + domainOrSolver + 'spec.html')" frameborder="0" style="width: 100%; height: 50vh"></iframe>
    </el-dialog>
  </div>
</template>

<script>
export default {
  props: {
    isSolver: {type: Boolean, default: false}
  },
  beforeCreate () {
    this.$store.commit('loadSelection')
  },
  data () {
    return {
      specDialogVisible: false,
      // templates: [
      //   {name: 'Domain', characteristics: {'Agent': 'MultiAgent', 'Concurrency': 'Parallel', 'Constraints': '(none)', 'Dynamics': 'Environment'}},
      //   {name: 'TestDomain', characteristics: {'Agent': 'SingleAgent', 'Concurrency': 'Sequential', 'Constraints': 'Constrained', 'Dynamics': 'UncertainTransitions'}}
      // ],
      // selection: {
      //   template: 'Domain',
      //   characteristics: {'Agent': 'MultiAgent', 'Concurrency': 'Sequential', 'Constraints': '(none)', 'Dynamics': 'Simulation'}
      // },
    }
  },
  computed: {
    domainOrSolver () {
      return (this.isSolver ? 'solver' : 'domain')
    },
    templates () {
      return this.$store.state.templates
    },
    selection () {
      return this.$store.state.selection
    },
    selectedTemplate () {
      return this.$store.getters.selectedTemplate
    },
  //   templateLevelIndex () {
  //     const levelIndex = {}
  //     this.characteristics.forEach(characteristic => {
  //       const index = characteristic.levels.indexOf(this.selectedTemplate.characteristics[characteristic.name])
  //       levelIndex[characteristic.name] = (index >= 0 ? index : 0)
  //     });
  //     return levelIndex
  //   }
  },
  methods: {
    updateSelection (selection, domainOrSolver) {
      this.$store.commit('updateSelection', {selection, domainOrSolver})
    },
    isFinetuned (characteristic, level, domainOrSolver) {
      return this.selectedTemplate[domainOrSolver].characteristics[characteristic] !== level
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
    'specDialogVisible' (newVal) {
      if (!newVal) {
        this.$store.commit('loadSelection')
      }
    }
//     'selection.template': {
//       immediate: true,
//       handler (newVal) {
//         this.selection.characteristics = Object.assign({}, this.selectedTemplate.characteristics)
//         //this.$emit('update:selectedItemIndex', (this.autoSelectFirst ? 0 : -1))
//       }
//     }
 }
}
</script>

<style scoped>
.tag {
  margin-bottom: 5px;
  margin-right: 5px;
}

.subtag {
  margin-right: 5px;
}
</style>