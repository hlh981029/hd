<template>
  <div class="sider-panel-container">
    <!-- <div class="sider-panel-subtitle">Source Image</div>
    <div class="sider-panel-image-box">
      <div
        v-if="label!==-1"
      >
        <div class="sider-panel-image-hover">
          <a-icon
            class="sider-panel-image-hover-icon"
            type="eye"
            :style="{ color: '#fafafa' }"
            @click="previewImage(sourceImageURL)"
          />
        </div>
        <img
          class="sider-panel-image"
          :src="sourceImageURL"
          alt="image"
        >
      </div>

      <div
        v-else
        class="sider-panel-image-placeholder"
      >
        <a-icon
          type="picture"
          class="sider-panel-image-placeholder-icon"
          theme="twoTone"
          twoToneColor="#d9d9d9"
        />
      </div>
    </div>
    <div class="sider-panel-divider"></div> -->
    <div class="sider-panel-subtitle">Class Activation Map</div>
    <div class="sider-panel-image-box">
      <div
        v-if="label!==-1"
      >
        <div class="sider-panel-image-hover">
          <a-icon
            class="sider-panel-image-hover-icon"
            type="eye"
            :style="{ color: '#fafafa' }"
            @click="previewImage(CAMURL)"
          />
        </div>
        <img
          class="sider-panel-image"
          :src="CAMURL"
          alt="image"
        >
      </div>
      <div
        v-else
        class="sider-panel-image-placeholder"
      >
        <a-icon
          type="picture"
          class="sider-panel-image-placeholder-icon"
          theme="twoTone"
          twoToneColor="#d9d9d9"
        />
      </div>
    </div>
    <div class="sider-panel-divider"></div>
    <div class="sider-panel-subtitle">Top 9 Pattern of Selected Channel</div>
    <a-spin :spinning="spinningKernel">
      <div
        class="sider-panel-kernel-container"
        v-if="kernelIndex!==-1"
      >
        <div
          class="sider-panel-kernel-image-box"
          v-for="n in 9"
          v-bind:key="n"
        >
          <div class="sider-panel-kernel-image-hover">
            <a-icon
              class="sider-panel-kernel-image-hover-icon"
              type="eye"
              :style="{ color: '#fafafa' }"
              @click="previewImage(backendURL+'pattern/'+layerIndex+'/'+kernelIndex+'_' + (n - 1) + '.png')"
            />
          </div>
          <img
            class="sider-panel-kernel-image"
            :src="backendURL+'pattern/'+layerIndex+'/'+kernelIndex+'_' + (n - 1) + '.png'"
            alt="image"
          >
        </div>
      </div>
      <div
        class="sider-panel-kernel-placeholder"
        v-else
      >
        <a-icon
          type="picture"
          class="sider-panel-kernel-placeholder-icon"
          theme="twoTone"
          twoToneColor="#d9d9d9"
        />
        <div>Please select a kernel first</div>
      </div>
    </a-spin>
    <a-modal
      title="Preview Image"
      v-model="visible"
      :footer="null"
    >
      <div class="sider-model-image">
        <img
          class="sider-preview-image"
          :src="imageURL"
          alt="image"
        >
      </div>
    </a-modal>
  </div>
</template>

<script>
export default {
  data() {
    return {
      filename: '',
      label: -1,
      layerName: '',
      layerIndex: -1,
      kernelIndex: -1,
      sourceImageURL: '',
      CAMURL: '',
      imageURL: '',
      visible: false,
      spinningKernel: false,
    };
  },
  methods:{
    updateCAM(filename, label) {
      this.filename = filename;
      this.label = label;
      this.sourceImageURL = this.backendURL + 'image/' + filename + '/' + filename;
      this.CAMURL = this.backendURL + 'image/' + filename + '/' + label + '_cam.jpg';
    },
    updateKernel(layerName, layerIndex, kernelIndex) {
      this.layerName = layerName;
      this.layerIndex = layerIndex;
      this.kernelIndex = kernelIndex;
      this.spinningKernel = false;
    },
    previewImage(URL) {
      this.imageURL = URL;
      this.visible = true;
    },
    reset() {
      this.filename = '';
      this.label = -1;
      this.layerName = '';
      this.layerIndex = -1;
      this.kernelIndex = -1;
      this.sourceImageURL = '';
      this.CAMURL = '';
      this.imageURL = '';
      this.visible = false;
    },
  },
};
</script>

<style>
.sider-panel-container {
  display: flex;
  flex-direction: column;
  overflow: auto;
  /* width: 298px; */
  height: 100%;
  width: 100%;
}
.sider-panel-image {
  max-width: 280px;
  max-height: 180px;
  height: auto;
  width: auto;
}
.sider-panel-image-placeholder {
  width: 280px;
  height: 180px;
  margin: 0 auto;
  background: #fafafa;
}
.sider-panel-subtitle {
  font-size: 18px;
  font-weight: 300;
  margin: 10px 0px;
}
.sider-panel-divider {
  height: 1px;
  border-bottom: 1px solid #d9d9d9;
  /* margin: 10px 0px; */
}
.sider-panel-image-box {
  align-self: center;
  margin-bottom: 20px;
  position: relative;
}
.sider-panel-kernel-container {
  display: flex;
  justify-content: center;
  width: 100%;
  flex-wrap: wrap;
}
.sider-panel-kernel-image {
  height: 85px;
  width: 85px;
}

.sider-panel-kernel-image-hover {
  opacity: 0;
  position: absolute;
  height: 85px;
  width: 85px;
  background-color: #333;
  transition: all 0.3s;
  z-index: -1;
}
.sider-panel-kernel-image-box {
  position: relative;
}
.sider-panel-kernel-image-box:hover .sider-panel-kernel-image-hover {
  z-index: 10;
  opacity: 0.7;
  transition: all 0.3s;
}
.sider-panel-kernel-image-hover-icon {
  font-size: 30px;
  margin-top: 27px;
  cursor: pointer;
}
.sider-panel-image-hover {
  opacity: 0;
  position: absolute;
  height: 100%;
  width: 100%;
  background-color: #333;
  transition: all 0.3s;
  z-index: -1;
}
.sider-panel-image-box:hover .sider-panel-image-hover {
  z-index: 10;
  opacity: 0.7;
  transition: all 0.3s;
}
.sider-panel-image-hover-icon {
  font-size: 30px;
  margin-top: 50px;
  cursor: pointer;
}
.sider-panel-image-placeholder-icon {
  font-size: 60px;
  margin-top: 35px;
}
.sider-panel-kernel-placeholder {
  width: 250px;
  height: 250px;
  margin: 0 auto;
  background: #fafafa;
}
.sider-panel-kernel-placeholder-icon {
  font-size: 60px;
  margin-top: 80px;
  margin-bottom: 10px;
}
.sider-model-image {
  display: flex;
  justify-content: center;
}
.sider-preview-image {
  max-width: 100%;
  max-height: 100%;
  min-width: 350px;
  min-height: 350px;

}
</style>