<template>
  <div class="file-uploader-outer-box">
    <div class="file-uploader-instruction">Here is a demo image, and you can select other demo images or your own image.</div>
    <div class="file-uploader-container">
      <div class="file-uploader-box">
        <a-upload-dragger
          name="file"
          accept="image/jpeg,image/png"
          :action="backendURL"
          :fileList="fileList"
          :showUploadList="false"
          :beforeUpload="beforeUpload"
          :disabled="uploading||isDemo"
          :customRequest="handleFileUpload"
        >
          <div
            v-if="imageURL"
            class="img-box"
          >
            <img
              :src="imageURL"
              alt="upload image"
            >
          </div>
          <div v-else>
            <p class="ant-upload-drag-icon">
              <a-icon type="picture" />
            </p>
            <p class="ant-upload-text">Click or drag file to this area to upload</p>
            <p class="ant-upload-hint">
              single jpg or png image, less than 2mb
            </p>
          </div>
        </a-upload-dragger>
      </div>
      <div class="button-box">
        <div>
          <div class="guide">Select other demos</div>
          <a-button
            block
            type="default"
            @click="handleShowOtherDemos"
          >
            Other Demos
          </a-button>
        </div>
        <div>
          <div class="guide">Select your own image</div>
          <a-button
            block
            type="default"
            @click="handleUpload"
          >
            Your Own Images
          </a-button>
        </div>
        <div>
          <div class="guide">
            Choose a class
          </div>
          <a-select
            class="class-select"
            :disabled="!success&&!isDemo"
            @change="handleSelectChange"
            :value="selectValue"
          >
            <a-select-option
              v-for="(class_name, index) in ClassName"
              :key="class_name"
              :value="index"
            >{{ index + '.' + class_name + ': ' + (isNaN(logits[index])?0:logits[index]).toFixed(2) }}</a-select-option>
          </a-select>
        </div>
        <!-- <div>
          <div class="guide">Click to show</div>
          <a-button
            block
            type="default"
            @click="handleShow"
            :disabled="!success&&!isDemo"
            :loading="waiting"
          >
            Show
          </a-button>
        </div> -->
      </div>
    </div>
    <a-modal
      title="Other Demos"
      v-model="visible"
      :footer="null"
      :width="700"
    >
      <div class="image-card-container">
        <img
          v-for="i in demoList"
          v-bind:key="i.filename"
          alt="example"
          class="image-card-image"
          :src="backendURL+'image/'+i.filename+'/'+i.filename"
          @click="setDemo(i.filename)"
        />
      </div>
    </a-modal>
  </div>
</template>

<script>
import ClassName from '../utils/ClassName.js';
import { demoList } from '../utils/demo.js';
export default {
  name: 'FileUploader',
  data() {
    return {
      imageURL: false,
      uploading: false,
      waiting: false,
      canPost: false,
      success: false,
      showLabel: -1,
      fileList: [],
      uploadFilename: '',
      ClassName, 
      logits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
      isDemo: true,
      demo: -1,
      selectValue: 'Class: Confidence',
      visible: false,
      demoList: [],
    };
  }, 
  methods: {
    reset() {
      this.selectValue = 'Class: Confidence';
      this.imageURL = '';
      this.logits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    },
    setDemo(filename) {
      this.isDemo = true;
      this.imageURL = this.backendURL + 'image/' + filename + '/' + filename;
      this.uploading = true;
      this.axios.post('/get_demo_label', {filename: filename})
        .then(response => {
          console.log(response);
          this.uploading = false;
          this.logits.splice(0, this.logits.length, ...response.data.result);
          this.selectValue = response.data.label;
          this.success = true;
          this.uploadFilename = filename;
          this.$emit('reset');
          this.$emit('upload-success', {
            filename: filename,
            result: response.data.result,
          });
          this.handleShow();
        })
        .catch(error => {
          console.log(error);
          this.uploading = false;
        });
      this.visible = false;
    },
    beforeUpload(file) {
      this.success = false;
      this.$emit('reset');
      const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
      if (!isJpgOrPng) {
        this.$message.error('You can only upload JPG/PNG file!');
      }
      const isLt2M = file.size / 1024 / 1024 < 2;
      if (!isLt2M) {
        this.$message.error('Image must smaller than 2MB!');
      }
      if (isJpgOrPng && isLt2M) {
        this.imageURL = URL.createObjectURL(file);
        this.canPost = true;
        this.fileList = [file];
        this.selectValue = 'Class: Confidence';
        this.isDemo = false;
        return true;
      } else {
        this.imageURL = false;
        this.fileList = [];
        this.canPost = false;
        return false;
      }
    },
    handleFileUpload() {
      const key = 'uploader';
      if (this.canPost) {
        this.$message.loading({ content: `Uploading`, key, duration: 0 });
        this.uploading = true;
        const formData = new FormData();
        formData.append('file', this.fileList[0]);
        this.axios.post('/get_label', formData)
          .then(response => {
            console.log(response);
            this.uploading = false;
            this.$message.success({ content: `Success`, key});
            this.$emit('upload-success', response.data);
            this.uploadFilename = response.data.filename;
            this.logits.splice(0, this.logits.length, ...response.data.result);
            this.success = true;
            this.selectValue = this.getMaxLabel(this.logits);
            console.log('predict label', this.selectValue);
            this.handleShow();
          })
          .catch(error => {
            console.log(error);
            this.uploading = false;
            this.$message.success({ content: `Failed`, key});
          });
      } else {
        this.$message.error('unknown error');
      }
    },
    handleSelectChange(value) {
      console.log('select: ', value);
      this.selectValue = value;
      this.handleShow();
    },
    handleShow() {
      this.showLabel = this.selectValue;
      this.waiting = true;
      this.axios.post('/get_kernel_images', {
        filename: this.uploadFilename,
        label: this.showLabel,
        indexes: [],
      }).then(response => {
        this.waiting = false;
        this.$emit('select-label', this.showLabel);
        this.$emit('show-success', {
          depth: 0,
          result: response.data.result,
          resultMas: response.data.result_mas,
          layerIndexes: [],
          channelIndexes: [],
        });
        console.log(response);
      }).catch(error => {
        console.log(error);
      });
    },
    handleUpload() {
      this.reset();
      this.$emit('reset');
      this.isDemo = false;
    },
    handleShowOtherDemos() {
      this.visible = true;
      this.demoList = demoList.slice(1, 7);
    },
    getMaxLabel(logits) {
      let label = 0;
      for (let i = 0; i < logits.length; i++) {
        if (isNaN(logits[i])) {
          console.log(i, 'isnan');
        }
        if (logits[i] > logits[label]) {
          label = i;
        }
      }
      return label;
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.file-uploader-container {
  display: flex;
  min-height: 300px;
}
.file-uploader-box {
  min-height: 300px;
  min-width: 300px;
  flex: 1 1 300px;
}
.button-box {
  /* max-width: 150px; */
  flex: 0 0 160px;
  margin-left: 20px;
  display: flex;
  flex-direction: column;
  justify-content: space-around;
}
img {
  max-width: 90%;
  max-height: 260px;
  width: auto;
  height: auto;
}
.guide {
  text-align: left;
  font-size: 15px;
}
.class-select {
  width: 100%
}
.file-uploader-instruction {
  text-align: left;
  font-size: 18px;
  margin-bottom: 10px;
  font-weight: 300;
}
.image-card-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-content: space-between;
  height: 360px;
  /* width: 660px;
  height: 500px; */
}
.image-card {
  width: 200px;
  height: 160px;
  text-align: center;
}
.image-card-image {
  object-fit: cover;
  width: 200px;
  height: 160px;
  margin-bottom: 10px;
  transition: all 0.3s;
}
.image-card-image:hover {
  cursor: pointer;
  transition: all 0.3s;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.55);
}

</style>
