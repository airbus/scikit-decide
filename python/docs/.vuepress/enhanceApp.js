// import Vuetify from './node_modules/vuetify/dist/vuetify.min.js' // './node_modules/vuetify'
// import './node_modules/vuetify/dist/vuetify.min.css'

import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';

export default ({
  Vue, // the version of Vue being used in the VuePress app
  options, // the options for the root Vue instance
  router, // the router instance for the app
  siteData // site metadata
}) => {
    Vue.use(ElementUI);
}