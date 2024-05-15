import Vue from 'vue';
import App from './App.vue';
import Axios from 'axios';

// -------------------- Ant Design -------------------- //
// import './plugins/ant-design-vue.js';
// import Antd from 'ant-design-vue';
// import 'ant-design-vue/dist/antd.css';
// Vue.use(Antd);
// spin, divider, icon, modal, switch, button, upload, 
import { Upload, Button, Spin, Divider, Icon, Switch, Modal, Select, Radio, message } from 'ant-design-vue';
Vue.prototype.$message = message;
const components = [Upload, Button, Spin, Divider, Icon, Switch, Modal, Select, Radio];
components.forEach(component => {
  Vue.use(component);
});

// -------------------- Element UI -------------------- //
// import ElementUI from 'element-ui';
// import 'element-ui/lib/theme-chalk/index.css';
// Vue.use(ElementUI);

Vue.prototype.axios = Axios;

if (process.env.NODE_ENV === 'development') {
  Vue.prototype.backendURL = 'http://mc.nankai.edu.cn/hdapi/';
} else {
  Vue.prototype.backendURL = '/hdapi/';
}
Axios.defaults.baseURL = Vue.prototype.backendURL;


Vue.config.productionTip = false;

new Vue({
  render: h => h(App),
}).$mount('#app');
