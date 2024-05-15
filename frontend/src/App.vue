<template>
  <div
    id="app"
  >
    <div class="header-container">
      Hierarchical Decomposition
    </div>
    <div
      class="body-container"
      ref="content"
    >
      <div
        class="content-container"
        id="content-container"
      >
        <file-uploader
          class="uploader"
          @upload-success="handleUploadSuccess"
          @show-success="handleShowSuccess"
          @select-label="handleSelectLabel"
          @reset="reset"
          ref="uploader"
        ></file-uploader>
        <!-- <transition name="fade"> -->  
        <div v-if="result[0].length!==0">
          <a-divider/>
          <a-spin :spinning="spinning[0]">
            <layer-panel
              class="panel conv5"
              :channel-list="result[0]"
              :channel-mas-list="resultMas[0]"
              :channel-indexes="channelIndexes[0]"
              :layer-indexes="layerIndexes[0]"
              layer-name="conv5-3"
              :layer-index="5"
              :filename="filename"
              :label="label"
              ref="conv5"
              :enhance="enhance"
              @show-success="handleShowSuccess"
              @show-kernel="handleShowKernel"
            ></layer-panel>
          </a-spin>
        </div>
        <!-- </transition> -->
        <div v-if="result[1].length!==0">
          <a-divider/>
          <a-spin :spinning="spinning[1]">
            <layer-panel
              class="panel conv4"
              :channel-list="result[1]"
              :channel-mas-list="resultMas[1]"
              :channel-indexes="channelIndexes[1]"
              :layer-indexes="layerIndexes[1]"
              layer-name="conv4-3"
              :layer-index="4"
              :filename="filename"
              :label="label"
              ref="conv4"
              :enhance="enhance"
              @show-success="handleShowSuccess"
              @show-kernel="handleShowKernel"
            ></layer-panel>
          </a-spin>
        </div>
        <div v-if="result[2].length!==0">
          <a-divider/>
          <a-spin :spinning="spinning[2]">
            <layer-panel
              class="panel conv3"
              :channel-list="result[2]"
              :channel-mas-list="resultMas[2]"
              :channel-indexes="channelIndexes[2]"
              :layer-indexes="layerIndexes[2]"
              layer-name="conv3-3"
              :layer-index="3"
              :filename="filename"
              :label="label"
              ref="conv3"
              :enhance="enhance"
              @show-success="handleShowSuccess"
              @show-kernel="handleShowKernel"
            ></layer-panel>
          </a-spin>
        </div>
        <div v-if="result[3].length!==0">
          <a-divider/>
          <a-spin :spinning="spinning[3]">

            <layer-panel
              class="panel conv2"
              :channel-list="result[3]"
              :channel-mas-list="resultMas[3]"
              :channel-indexes="channelIndexes[3]"
              :layer-indexes="layerIndexes[3]"
              layer-name="conv2-2"
              :layer-index="2"
              :filename="filename"
              :label="label"
              ref="conv2"
              :enhance="enhance"
              @show-success="handleShowSuccess"
              @show-kernel="handleShowKernel"
            ></layer-panel>
          </a-spin>
        </div>
        <div v-if="result[4].length!==0">
          <a-divider/>
          <a-spin :spinning="spinning[4]">
            <layer-panel
              class="panel conv1"
              :channel-list="result[4]"
              :channel-mas-list="resultMas[4]"
              :channel-indexes="channelIndexes[4]"
              :layer-indexes="layerIndexes[4]"
              layer-name="conv1-2"
              :layer-index="1"
              :filename="filename"
              :label="label"
              ref="conv1"
              :enhance="enhance"
              @show-success="handleShowSuccess"
              @show-kernel="handleShowKernel"
            ></layer-panel>
          </a-spin>
        </div>
      </div>
      <div
        class="sider-container"
        id="fixed"
        :style="{right: rightOffset+'px'}"
      >
        <sider-panel
          :filename="filename"
          ref="sider"
        >
        </sider-panel>
      </div>
    </div>
  </div>
</template>

<script>
import FileUploader from './components/FileUploader.vue';
import LayerPanel from './components/LayerPanel.vue';
import SiderPanel from './components/SiderPanel.vue';
import { demoList } from './utils/demo.js';
export default {
  name: 'app',
  created() {
    window.addEventListener('scroll', this.handleScroll);
    window.addEventListener('resize', this.handleResize);
    // this.demoIndex = 0;
    // const demo = demoList[this.demoIndex];
    // this.filename = this.demo.filename;
    // this.label = this.demo.label;
    // this.logits = this.demo.logits;
  },
  mounted() {
    this.$refs.uploader.setDemo(demoList[0].filename);
    this.handleResize();
  },
  data() {
    return {
      filename: '',
      result: [[], [], [], [], [], []],
      resultMas: [[], [], [], [], [], []],
      channelIndexes: [[], [], [], [], []],
      layerIndexes: [[], [], [], [], []],
      logits: [],
      label: -1,
      rightOffset: 0,
      spinning: [false, false, false, false, false, false],
      spinningTime: 0,
      spinningDelay: 0,
      demoList: demoList,
      demoIndex: -1,
      demo: -1,
      enhance: true,
    };
  },
  components: {
    FileUploader,
    LayerPanel,
    SiderPanel,
  },
  methods: {
    reset() {
      this.filename = '';
      this.result.splice(0, 5, ...[[], [], [], [], []]);
      this.resultMas.splice(0, 5, ...[[], [], [], [], []]);
      this.channelIndexes.splice(0, 5, ...[[], [], [], [], []]);
      this.layerIndexes.splice(0, 5, ... [[], [], [], [], []]);
      this.logits.splice(0, this.logits.length);
      this.label = -1;
      this.$refs.sider.reset();
    },
    handleUploadSuccess(data) {
      console.log(data);
      this.filename = data.filename;
      this.logits = data.result;
    },
    handleShowSuccess(data) {
      const self = this;
      console.log('handleShowSuccess', data);
      if (data.depth === 0) {
        this.$refs.sider.updateCAM(this.filename, this.label);
      }
      self.result[data.depth].splice(0, 5, ...data.result.slice(0, 5));
      self.resultMas[data.depth].splice(0, 5, ...data.resultMas.slice(0, 5));
      self.channelIndexes[data.depth].splice(0, data.depth, ...data.channelIndexes);
      self.layerIndexes[data.depth].splice(0, data.depth, ...data.layerIndexes);
      self.spinning.splice(data.depth, 1, false);
      if (data.depth !== 0) {
        self.$refs.sider.updateKernel(data.layerName, data.layerIndex, data.kernelIndex);
      }
      if (data.depth === 0 && this.$refs.conv5 !== undefined) {
        this.$refs.conv5.reset();
        this.result.splice(1, 4, ...[[], [], [], []]);
        this.resultMas.splice(1, 4, ...[[], [], [], []]);
        this.channelIndexes.splice(1, 4, ...[[], [], [], []]);
        this.layerIndexes.splice(1, 4, ...[[], [], [], []]);
      }
    },
    handleSelectLabel(label) {
      this.label = label;
    },
    handleScroll() {
      const scrollOffset = document.body.scrollLeft || document.documentElement.scrollLeft;
      if (scrollOffset !== 0) {
        this.rightOffset = (document.body.scrollLeft || document.documentElement.scrollLeft) + document.body.offsetWidth - this.$refs.content.offsetWidth - 300;
      }
    },
    handleResize() {
      const contentWidth = this.$refs.content.offsetWidth;
      if (contentWidth >= 950) {
        this.rightOffset = (this.$refs.content.offsetWidth - 950) / 2;
        console.log(this.rightOffset);
      } else {
        this.rightOffset = document.body.offsetWidth - this.$refs.content.offsetWidth - 300;
      }
    },
    handleShowKernel(data) {
      console.log('handleShowKernel');
      const nextLayerIndex = 6 - data.layerIndex;
      console.log(6 - data.layerIndex);
      this.spinning.splice(nextLayerIndex, 1, true);
      this.$refs.sider.spinningKernel = true;
      if (this.result[nextLayerIndex].length === 9) {
        this.$refs['conv' + (data.layerIndex - 1)].reset();
      }
      for (let i = 7 - data.layerIndex; i < 5; i++) {
        this.result.splice(i, 1, []);
        this.channelIndexes.splice(i, 1, []);
        this.layerIndexes.splice(i, 1, []);
      }
      this.spinningTime = Date.now();
      if (data.layerIndex === 1) {
        setTimeout(() => {
          this.$refs.sider.updateKernel(data.layerName, data.layerIndex, data.kernelIndex);
        }, this.spinningDelay);
      }
      // this.first = false;
    },
  },
};
</script>

<style>
#app {
  font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  min-width: 850px;
}
.uploader {
  margin: 15px 30px 20px;
}
.panel {
  padding: 0px 30px 0px 15px;
}
.content-container {
  flex: 1 1 550px;
  max-width: 950px;
}
.sider-container {
  position: fixed;
  height: calc(100vh - 70px);
  width: 300px;
  border-left: 1px solid #d9d9d9;
  z-index: 1;
  /* background: #aaa */
}
.header-container {
  position: fixed;
  top: 0px;
  font-size: 40px;
  height: 70px;
  width: 100vw;
  border-bottom: 1px solid #d9d9d9;
  font-weight: 300;
  line-height: 70px;
  z-index: 2;
  background: #fff;
}
.body-container {
  display: flex;
  margin-right: 300px;
  margin-top: 70px;
  justify-content: center;
}
</style>
