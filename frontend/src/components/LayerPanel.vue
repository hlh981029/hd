<template>
  <div class="layer-panel-container">
    
    <!-- <div class="layer-panel-header">
      <div class="layer-panel-title">
        <span class="layer-panel-title-content">Top 5 of {{layerName}}</span> <br/> {{layerIndex!==5?'From kernel '+channelIndexes.join(' > '): ''}}
      </div>
      <div class="layer-panel-button-box">
        <div class="image-enhance-box"> -->
    <!-- <span
            class="guide"
          >Show Visualization</span> -->
    <!-- <a-switch
            defaultChecked
            v-model="enhance"
          ></a-switch> -->
    <!-- <a-radio-group
            :value="view" 
            @change="handleViewChange"
          >
            <a-radio-button value="cam">
              Activation Map
            </a-radio-button>
            <a-radio-button value="vis">
              Visualization
            </a-radio-button>
          </a-radio-group>
        </div>
        <a-button @click="handlePreview">Preview Image</a-button>
        <a-button
          @click="handleGetDemo"
          v-if="layerIndex===1"
        >Get Demo</a-button>
      </div>
    </div> -->
    <div class="layer-panel-content">

    
      <div class="layer-panel-title text-v-lr">
        <span class="layer-panel-title-content text-v-lr">Top 5 of {{layerName}}</span>
      <!-- <br/> -->
      <!-- {{layerIndex!==5?'From kernel '+channelIndexes.join(' > '): ''}} -->
      </div>
      <div class="image-list-container">
        <div
          @click="handleRadioChange(index)"
          :class="value===index?'image-radio-box-selected':'image-radio-box'"
          v-for="(channel, index) in channelList"
          v-bind:key="channel"
        >
          <div
            :class="value===index?'image-box-divider-selected':'image-box-divider'"
          >
            <!-- <img
            class="image"
            :src="backendURL+'image/'+filename+'/'+label+'_'+layerIndex+'_'+[...channelIndexes, channel].join('_')+(enhance?'_cam.jpg':'.jpg')"
            alt="image"
          > -->
            <img
              class="image-top"
              :src="backendURL+'image/'+filename+'/'+label+'_'+layerIndex+'_'+[...channelIndexes, channel].join('_')+'_cam.jpg'"
              alt="image"
            >
            <img
              class="image-bottom"
              :src="backendURL+'image/'+filename+'/'+label+'_'+layerIndex+'_'+[...channelIndexes, channel].join('_')+'.jpg'"
              alt="image"
            >
          </div>
          <div class="image-title">
            Channel {{channel}}: {{Math.round(channelMasList[index] * 10000) / 100 + "%"}}
          </div>
        </div>
        <div class="image-radio-box-placeholder"></div>
      </div>
    </div>
    <a-modal
      title="Preview Image"
      v-model="visible"
      :footer="null"
    >
      <div class="model-image">
        <img
          class="preview-image"
          :src="imageURL"
          alt="image"
        >
      </div>
    </a-modal>
  </div>
</template>

<script>
export default {
  name: 'LayerPanel',
  props:{
    channelList: Array,
    channelMasList: Array,
    channelIndexes: Array,
    layerIndexes: Array,
    layerName: String,
    layerIndex: Number,
    filename: String,
    label: Number,
    isDemo: Boolean,
  },
  data() {
    return {
      value: -1,
      visible: false,
      imageURL: '',
      waiting: false,
      enhance: true,
      view: 'cam',
    };
  },
  mounted() {
    console.log('layer-panel-' + this.layerName + ' Mounted');
  },
  methods: {
    handleRadioChange(index) {
      console.log('handleRadioChange: ', index);
      this.value = index;
      this.$emit('show-kernel', {
        layerName: this.layerName,
        layerIndex: this.layerIndex,
        kernelIndex: this.channelList[index],
      });
      if (this.layerIndex !== 1) {
        this.handleShowNext(index);
      }
    },
    handleViewChange(e) {
      console.log(e);
      this.view = e.target.value;
      this.enhance = this.view === 'cam';
      // if (this.view == 'cam') {
      //   this.enhance = true;
      // } else {
      //   this.enhance = false;
      // }
    },
    handleShowNext(index) {
      this.waiting = true;
      this.axios.post('/get_kernel_images', {
        filename: this.filename,
        label: this.label,
        indexes: [...this.layerIndexes, index],
      }).then(response => {
        this.$emit('show-success', {
          depth: 6 - this.layerIndex,
          layerIndexes: [...this.layerIndexes, index],
          channelIndexes: [...this.channelIndexes, this.channelList[index]],
          result: response.data.result,
          resultMas: response.data.result_mas,
          layerName: this.layerName,
          layerIndex: this.layerIndex,
          kernelIndex: this.channelList[index],
        });
        console.log('/get_kernel_images', response);
        this.waiting = false;
      }).catch(error => {
        console.log(error);
      });
    },
    handlePreview() {
      if (this.value === -1) {
        this.$message.info('Please select a kernel first.');
      } else {
        this.imageURL = this.backendURL + 'image/' + this.filename + '/' +
                        this.label + '_' + this.layerIndex + '_' +
                        [...this.channelIndexes, this.channelList[this.value]].join('_') +
                        (this.enhance ? '_cam.jpg' : '.jpg');
        this.visible = true;
      }
    },
    handleGetDemo() {
      this.waiting = true;
      this.axios.post('/gen_demo', {
        filename: this.filename,
        label: this.label,
        indexes: this.layerIndexes,
      }).then(response => {
        if (response.data.result === 0) {
          this.$message.success('success');
          window.open(this.backendURL + 'image/' + this.filename + '/' + this.filename + '.pdf');
        } else {
          this.$message.error('error');
        }
        this.waiting = false;
      }).catch(error => {
        console.log(error);
      });
    },
    reset() {
      this.value = -1;
    },
    select(value) {
      this.value = value;
    },
  },
};
</script>

<style>
.layer-panel-header{
  padding-bottom: 10px;
  display: flex;
  justify-content: space-between;
}
.layer-panel-title {
  text-align: left;
  font-size: 16px;
}
.image-list-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
}
.image-radio-box {
  cursor: pointer;
  width: 150px;
  border: #d9d9d9 solid 1px;
  
  border-radius: 4px;
  margin-bottom: 10px;
  overflow: hidden;
  /* height: 170px; */
  transition: all 0.3s;
}
.image-radio-box-selected {
  cursor: pointer;
  width: 150px;
  border: #1890ff solid 1px;
  color: #1890ff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  margin-bottom: 10px;
  overflow: hidden;
  /* height: 170px; */
  transition: all 0.3s;
}
.image-radio-box:hover {
  cursor: pointer;
  width: 150px;
  border: #1890ff solid 1px;
  color: #1890ff;
  border-radius: 4px;
  margin-bottom: 10px;
  overflow: hidden;
  transition: all 0.3s;
}
.image-radio-box-placeholder {
  width: 150px;
  /* height: 170px; */
}
.image-box-divider {
  border-bottom: #d9d9d9 solid 1px;
  transition: all 0.3s;
}
.image-radio-box:hover .image-box-divider {
  border-bottom: #1890ff solid 1px;
  transition: all 0.3s;
}
.image-box-divider-selected {
  border-bottom: #1890ff solid 1px;
  transition: all 0.5s;
}
.image {
  width: 130px;
  height: 130px;
  margin: 10px;
}
.image-top {
  width: 130px;
  height: 130px;
  margin: 10px 10px 5px;
}
.image-bottom {
  width: 130px;
  height: 130px;
  margin: 5px 10px 10px;
}
.image-title {
  margin: 7px;
}
.preview-image {
  width: 300px;
  height: 300px;
}
.model-image {
  display: flex;
  justify-content: center;
}
.layer-panel-title-content {
  font-weight: 300;
  font-size: 24px;
}
.image-enhance-box {
  display: inline-flex;
  justify-content: space-between;
  width: 250px;
  margin-right: 20px;
}
.text-v-lr {
  writing-mode: vertical-lr;
  margin-right: 10px;
}
.layer-panel-title-v {
  writing-mode: vertical-lr;
}
.layer-panel-content {
  display: flex;
}
</style>
