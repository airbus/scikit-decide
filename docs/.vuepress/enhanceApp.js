// import Vuetify from './node_modules/vuetify/dist/vuetify.min.js' // './node_modules/vuetify'
// import './node_modules/vuetify/dist/vuetify.min.css'

import Vuex from 'vuex'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'

import default_state from './_state.json'

export default ({
  Vue, // the version of Vue being used in the VuePress app
  options, // the options for the root Vue instance
  router, // the router instance for the app
  siteData // site metadata
}) => {
    Vue.use(Vuex)
    Vue.use(ElementUI)

    const store = new Vuex.Store({
      state: default_state,
      getters: {
        selectedTemplate (state) {
          let selectedTemplate = {}
          const elements = ['domain', 'solver']
          elements.forEach(element => {
            const templateIndex = state.templates[element].findIndex(t => t.name == state.selection[element].template)
            selectedTemplate[element] = state.templates[element][templateIndex]
          })
          return selectedTemplate
        },
        domainTypes (state) {
          const levels = Object.values(state.selection['domain'].characteristics).filter(c => c !== '(none)')
          return Object.assign({}, ...levels.map(l => state.types['domain'][l]))
        }
      },
      mutations: {
        loadSelection (state) {
          try {  // avoid error when rendering server-side (no access to localStorage)
            const rawStoredSelection = localStorage.getItem('selection')
            if (rawStoredSelection) {
              // Replace the state selection with the stored item
              const storedSelection = JSON.parse(rawStoredSelection)
              state.selection = {...state.selection, ...storedSelection}
            }
          }
          catch(err) {
            console.log(err.message)
          }
        },
        updateSelection (state, {selection, domainOrSolver}) {
          state.selection[domainOrSolver] = {...state.selection[domainOrSolver], ...selection}
          localStorage.setItem('selection', JSON.stringify(state.selection))
        }
      }
    })
    Vue.mixin({store: store})  //, beforeCreate: () => store.commit('loadSelection')})
}