/***************** MNIST DATA ********************/

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

let NUM_TRAIN_ELEMENTS = 60000;
let NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
            new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize);
          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages =
        this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
        this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
        this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
        batchSize, [this.trainImages, this.trainLabels], () => {
          this.shuffledTrainIndex =
              (this.shuffledTrainIndex + 1) % this.trainIndices.length;
          return this.trainIndices[this.shuffledTrainIndex];
        });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image =
          data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label =
          data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels};
  }
}







/*********** VISUALIZATION ************/
let exampleTexts = ['Example', 'my Model']
let modelLayers = [];
let layerNames = ['Conv', 'Acti','Pool', 'Linear'];
const activationLayers = ['ReLU', 'Tanh', 'Sigmoid'];
let usedActivationLayers = [];
const filterSizes = [3, 5];
const strides = [1, 2];
const paddings = [0, 1];
const tfPaddings = ['valid', 'same'];
const channels = [4, 6, 8, 16, 32, 64, 128];
const learningRates = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1];
const poolSizes = [2, 3, 4];
let numLayer = 0;
let numAddBtn = 1;
let addBtns = [0];
let stopLearning = false, pauseLearning = false;
let epochCurr = 0;





function channelCheck(prevLayerDiv){
    while(!prevLayerDiv.classList.contains('newLayer') ||
        d3.select(prevLayerDiv).select('.layerTitleBtn').node().innerText != layerNames[0]){
        if(prevLayerDiv.previousElementSibling == null){
            return 1;
        } else{
            prevLayerDiv = prevLayerDiv.previousElementSibling;
        }
    }
    console.log(d3.select(prevLayerDiv).node())
    return Number(d3.select(prevLayerDiv).select('div > .chanOut').select('.convChannel').property('value'));
}
function translate(x, y) {
    return `translate(${x}, ${y})`
}
const EXPANDSVG = 3;
const Duration = 500;

function showSizes(svg, inp, outp, filt, str, padd){
    console.log('input: ' + inp + 'output: ' + outp + 'filter:' + filt + 'stride: ' + str + 'padd: ' + padd)
    const entireSquare = svg.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 28 * EXPANDSVG)
        .attr('height', 28 * EXPANDSVG)
        .attr('class', 'entireSquare')
        
    const inputSquare = svg.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', inp * EXPANDSVG)
        .attr('height', inp * EXPANDSVG)
        .attr('class', 'inputSquare')
        
    const padSquare = svg.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', (outp + 2 * padd) * EXPANDSVG)
        .attr('height', (outp + 2 * padd) * EXPANDSVG)
        .attr('class', 'padSquare')
    const outputSquare = svg.append('rect')
        .attr('x', padd * EXPANDSVG)
        .attr('y', padd * EXPANDSVG)
        .attr('width', (outp) * EXPANDSVG)
        .attr('height', (outp) * EXPANDSVG)
        .attr('class', 'outputSquare')
    const filterSquare = svg.append('rect')
        .attr('x', padd * EXPANDSVG)
        .attr('y', padd * EXPANDSVG)
        .attr('width', filt * EXPANDSVG)
        .attr('height', filt * EXPANDSVG)
        .attr('class', 'filterSquare')
    return;
}
const batchChange = d3.select('#batchSize').on('change', function(){
    d3.select('.batchSize').text(d3.select(this).property('value'))
})
d3.select('.exampleGroup').selectAll('button').property('disabled', true)
d3.select('.exampleGroup').selectAll('select').property('disabled', true)
const [btnSvgWidth, btnSvgHeight] = [300, 100];
const [iconWidth, iconHeight] = [50, 50];
const playClick = async function () {
    
    if(this.innerHTML == '<i class="fas fa-play-circle fa-3x"></i>'){   //play
        this.innerHTML = '<i class="fas fa-pause-circle fa-3x"></i>';
        d3.select('.modelSpace').selectAll('button').property('disabled', true)
        d3.select('.modelSpace').selectAll('select').property('disabled', true)
        d3.select('.betweenSpaces').selectAll('select').property('disabled', true)
        d3.select('.codeSpace').selectAll('select').property('disabled', true)
        stopLearning = false;
        pauseLearning = false;
        old_history = history;
        history = [];
        run();
    }
    else if(this.innerHTML == '<i class="fas fa-pause-circle fa-3x"></i>'){  //pause
        this.innerHTML = '<i class="fas fa-play-circle fa-3x"></i>';
        d3.select('.modelSpace').selectAll('button').property('disabled', true)
        d3.select('.modelSpace').selectAll('select').property('disabled', true)
        d3.select('.betweenSpaces').selectAll('select').property('disabled', true)
        d3.select('.codeSpace').selectAll('select').property('disabled', true)
        stopLearning = false;
        pauseLearning = true;
        //pause learning
    }
    else{   //stop
        this.previousElementSibling.innerHTML = '<i class="fas fa-play-circle fa-3x"></i>';
        //stop learning
        d3.select('.modelSpace').selectAll('button').property('disabled', false)
        d3.select('.modelSpace').selectAll('select').property('disabled', false)
        d3.select('.betweenSpaces').selectAll('select').property('disabled', false)
        d3.select('.codeSpace').selectAll('select').property('disabled', false)
        stopLearning = true;
        pauseLearning = false;
        epochCurr = 0;
        d3.select('#epochCurr').text(epochCurr)
        old_history = JSON.parse(JSON.stringify(history));
    }
}
const btnTopSvg = d3.selectAll('.btnTop')
const btnLearnFunc = btnTopSvg.selectAll('.btnLearning').on('click', playClick);
const btnExample = btnTopSvg.select('#Example').on('click', function(){
    const state = this.innerText;
    if(state == exampleTexts[0]){
        this.innerText = exampleTexts[1];
        d3.select('.addGroup').style('visibility', 'hidden')
        d3.select('.exampleGroup').style('visibility', 'visible')
    } else{
        this.innerText = exampleTexts[0];
        d3.select('.addGroup').style('visibility', 'visible')
        d3.select('.exampleGroup').style('visibility', 'hidden')
    }
})

const btnSavePy = btnTopSvg.select('#savePy').on('click', function(){

})
const dataSpace = d3.select('.modelSpace').select('.data')
const modelSpace = d3.select('.modelSpace').select('.model')
const resultSpace = d3.select('.modelSpace').select('.result')

const dataRect = dataSpace.append('rect')
    .attr('width', 100)
    .attr('height', 100)
    .attr('fill', 'white')
    .style('stroke', d3.rgb(192, 192, 192))
    .style('stroke-width', '2px')
    
const dataText = dataSpace.append('text')
    .text('Data')
    .attr('x', 30)
    .attr('y', 20)
    
const dataSizeText = dataSpace.append('text')
    .text('28 * 28')
    .attr('x', 25)
    .attr('y', 90)
// const dataPicture = dataSpace.append('canvas')
//     .attr('x', 20)
//     .attr('y', 50)
//     .attr('width', 28)
//     .attr('height', 28)
const testSizeDD = d3.select('.dataForm').append('svg')
    .attr('width', 300)
    .attr('height', 50)
    .append('text')
    .attr('x', 0)
    .attr('y', 15)
    .attr('width', 30)
    .attr('height', 30)
    .text('Test data: 5000')
    .style('color', 'black')
const trainSizeDD = d3.select('#trainData')
    .on('change', function(d){
        let selectedOption = d3.select(this).property('value')
        testSizeDD
            .text('Test data: ' + (65000 - selectedOption))
            
        console.log('data changed')
        NUM_TRAIN_ELEMENTS = Number(d3.select(this).property('value'))
        NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
})

let currLib = 'Pytorch'

const changeChannel = function(){
    const channelVal = d3.select(this).property('value')
    let layerDiv = this.parentNode.parentNode.parentNode;
    while((layerDiv = layerDiv.nextElementSibling) != null){
        if(layerDiv.classList.contains('newLayer') && 
            d3.select(layerDiv).select('div > .layerTitleBtn').node().innerText == layerNames[0]){
                d3.select(layerDiv).select('div > .chanIn').text('Ch in: ' + channelVal)
                return;
            }
    }
}
const changeNextInputSize = function(grandParent){
    if(grandParent === null)    return;
    let prevGP = grandParent.previousElementSibling;
    console.log('prevGP: ', prevGP);
    console.log('grandParent: ', grandParent);
    while(grandParent.nextElementSibling !== null){
        if(grandParent.nextElementSibling.classList.contains('newLayer'))   break;
        else{
            grandParent = grandParent.nextElementSibling;
        }
    }
    while(prevGP !== null){
        if(prevGP.classList.contains('newLayer'))   break;
        else{
            prevGP = prevGP.previousElementSibling;
        }
    }
    const layerDiv = d3.select(grandParent).select('div')
    if(grandParent !== null){
        
        if(prevGP !== null){
            if(prevGP.className == 'addLayerDiv newLayer'){
                layerDiv.attr('id', d3.select(prevGP).select('.outputTextField').node().innerText)
                console.log('layerDiv id: ' + layerDiv.property('id'))
                layerDiv.select('.input')
                .text('Input: ' + Number(layerDiv.property('id')))
            }
        }
        else{
            layerDiv.attr('id', 28)
        } 
    console.log('layerDiv :', layerDiv.node())
    if(layerDiv.node().classList.contains('layerVis')){
        const layerCategory = layerDiv.select('.layerTitleBtn').node().innerText;
        const inputSize = Number(layerDiv.property('id'))
        layerDiv.select('.input').text('Input: ' + inputSize )
        let outputSize;
        if(layerCategory == layerNames[0]){
            
            let f = layerDiv.select('.convFilter').property('value')
            let p = layerDiv.select('.convPadding').property('value')
            let s = layerDiv.select('.convStride').property('value')
            outputSize = Math.round((inputSize + 2 * p - f) / s) + 1
            
            layerDiv.select('svg').select('.padSquare')
                .transition()
                .duration(Duration)
                .attr('width', (outputSize + 2 * p) * EXPANDSVG)
                .attr('height', (outputSize + 2 * p) * EXPANDSVG)
            layerDiv.select('svg').select('.outputSquare')
                .transition()
                .duration(Duration)
                .attr('width', outputSize * EXPANDSVG)
                .attr('height', outputSize * EXPANDSVG)

            layerDiv.select('.outputTextField').text(outputSize)
        } else if(layerCategory == layerNames[1]){
            layerDiv.select('.outputTextField').text(inputSize)
        } else if(layerCategory == layerNames[2]){
            let f = layerDiv.select('.poolFilter').property('value')
            let p = layerDiv.select('.poolPadding').property('value')
            let s = f
            outputSize = Math.round((inputSize + 2 * p - f) / s) + 1

            layerDiv.select('svg').select('.padSquare')
                .transition()
                .duration(Duration)
                .attr('width', (outputSize + 2 * p) * EXPANDSVG)
                .attr('height', (outputSize + 2 * p) * EXPANDSVG)
            layerDiv.select('svg').select('.outputSquare')
                .transition()
                .duration(Duration)
                .attr('width', outputSize * EXPANDSVG)
                .attr('height', outputSize * EXPANDSVG)
            layerDiv.select('.outputTextField').text(outputSize)
        } else if(layerCategory == layerNames[3]){

        }
        changeNextInputSize(grandParent.nextElementSibling)
        }
    }
}
const closeFunc = function(){
    const grandParent = this.parentNode.parentNode;
    const nextGP = grandParent.nextElementSibling;
    const prevGP = grandParent.previousElementSibling;
    
    d3.select(this.parentNode.parentNode).remove()
    console.log('nextGP: ' + nextGP)
    changeNextInputSize(nextGP);
}
const makeLayer = function (layerDiv, prevSibling, value) {
    console.log('Make layer')
    layerDiv.attr('class', 'layerVis'.concat(' ', value))
    console.log('value: ' + value)
    layerDiv.append('button')
        .html(closeBtnHtml)
        .attr('class', 'closeBtn')
        .on('click', closeFunc)
    const layerTitleBtn = layerDiv.append('button')
        .attr('class', 'layerTitleBtn')
        .text(value)
        .style('float', 'left')
    layerTitleBtn.on('click', function(){
        //closeFunc
        d3.select(this.parentNode.parentNode).remove()
        //addLayerFunc
        d3.select(this).addLayerFunc();
    })
    let inputSize = 28;
    let outputSize;
    while(prevSibling != null && !prevSibling.classList.contains('newLayer')){
        prevSibling = prevSibling.previousElementSibling;
    }
    if(prevSibling != null && prevSibling.classList.contains('newLayer'))
        inputSize = Number(d3.select(prevSibling).select('.layerVis').select('.outputTextField').node().innerText);
    layerDiv.attr('id', inputSize)
    console.log(layerDiv.property('id'))
    
    if(value == layerNames[0]){
        const layerShow = layerDiv.append('svg')
            .attr('width', 90)
            .attr('height', 90)
        const labelInput = layerDiv.append('label')
            .text('Input: ' + Number(layerDiv.property('id')))
            .style('float', 'left')
            .attr('class', 'input')
        const labelFilter = layerDiv.append('label')
            .text('Filter: ')
            .style('float', 'left')
            .attr('class', 'filter')
        const dropdownFilter = layerDiv.append('select')
            .attr('class', 'convFilter')
        dropdownFilter.selectAll('option').data(filterSizes)
            .enter().append('option')
            .attr('value', d => d)
            .html(d =>{ return d + '*' + d})
        dropdownFilter.on('change', function(){
            f = dropdownFilter.property('value')
            outputSize = Math.round((Number(layerDiv.property('id')) + 2 * p - f) / s) + 1
            outputTextField.text(outputSize)
            layerShow.select('.filterSquare')
                .transition()
                .duration(Duration)
                .attr('width', f * EXPANDSVG)
                .attr('height', f * EXPANDSVG)
            layerShow.select('.padSquare')
                .transition()
                .duration(Duration)
                .attr('width', (outputSize + 2 * p) * EXPANDSVG)
                .attr('height', (outputSize + 2 * p) * EXPANDSVG)
            layerShow.select('.outputSquare')
                .transition()
                .duration(Duration)
                .attr('width', outputSize * EXPANDSVG)
                .attr('height', outputSize * EXPANDSVG)
            const grandParent = this.parentNode.parentNode
            changeNextInputSize(grandParent.nextElementSibling)
        })
        const labelStride = layerDiv.append('label')
            .text('Stride: ')
            .style('float', 'left')
            .attr('class', 'stride')
        const dropdownStride = layerDiv.append('select')
            .attr('class', 'convStride')
        dropdownStride.selectAll('option').data(strides).enter().append('option')
            .attr('value', d => d)
            .html(d => d)
        dropdownStride.on('change', function(){
            s = dropdownStride.property('value')
            outputSize = Math.round((Number(layerDiv.property('id')) + 2 * p - f) / s) + 1
            outputTextField.text(outputSize)
            
            layerShow.select('.outputSquare')
                .transition()
                .duration(Duration)
                .attr('width', outputSize * EXPANDSVG)
                .attr('height', outputSize * EXPANDSVG)
            
            layerShow.select('.padSquare')
                .transition()
                .duration(Duration)
                .attr('width', (outputSize + 2 * p) * EXPANDSVG)
                .attr('height', (outputSize + 2 * p) * EXPANDSVG)
            const grandParent = this.parentNode.parentNode
            changeNextInputSize(grandParent.nextElementSibling)
        })
        const labelPadding = layerDiv.append('label')
            .text('Padding: ')
            .style('float', 'left')
            .attr('class', 'pad')
        const dropdownPadding = layerDiv.append('select')
            .attr('class', 'convPadding')
        dropdownPadding.selectAll('option').data(paddings).enter().append('option')
            .attr('value', d => d)
            .html(d => d)
        dropdownPadding.on('change', function(){
            p = dropdownPadding.property('value')
            outputSize = Math.round((Number(layerDiv.property('id')) + 2 * p - f) / s) + 1
            outputTextField.text(outputSize)
            layerShow.select('.padSquare')
                .transition()
                .duration(Duration)
                .attr('width', (outputSize + 2 * p) * EXPANDSVG)
                .attr('height', (outputSize + 2 * p) * EXPANDSVG)
            
            layerShow.select('.outputSquare')
                .transition()
                .duration(Duration)
                .attr('x', p * EXPANDSVG)
                .attr('y', p * EXPANDSVG)
                .attr('width', outputSize * EXPANDSVG)
                .attr('height', outputSize * EXPANDSVG)
            
            layerShow.select('.filterSquare')
                .transition()
                .duration(Duration)
                .attr('x', p * EXPANDSVG)
                .attr('y', p * EXPANDSVG)
            
            const grandParent = this.parentNode.parentNode;
            changeNextInputSize(grandParent.nextElementSibling);
        })
        let chanIn = channelCheck(layerDiv.node().parentNode.previousElementSibling); //channel only changes by conv
        const labelChanIn = layerDiv.append('label')
            .text('Ch in: ' + chanIn)
            .attr('class', 'chanIn')
            .style('float', 'left')
        const labelChanOut = layerDiv.append('label')
            .text('Ch out: ')
            .attr('class', 'chanOut')
            .style('float', 'left')
        const dropdownChannel = labelChanOut.append('select')
            .attr('class', 'convChannel')
        dropdownChannel.selectAll('option').data(channels)
            .enter().append('option')
            .attr('value', d => d)
            .html(d => d)
        dropdownChannel.on('change', changeChannel)
        let f = dropdownFilter.property('value')
        let p = dropdownPadding.property('value')
        let s = dropdownStride.property('value')
        outputSize = Math.round((Number(layerDiv.property('id')) + 2 * p - f) / s) + 1
        console.log('inputSize: ' + Number(layerDiv.property('id')) + ', f: ' + f + ", p: " + p + ', s: ' + s + ', outputSize:' + outputSize)

        const labelOutput = layerDiv.append('label')
            .text('Output: ')
            .style('float', 'left')
        const outputTextField = layerDiv.append('text')
            .attr('class', 'outputTextField')
            .text(outputSize)
        showSizes(layerShow, Number(layerDiv.property('id')), outputSize, dropdownFilter.property('value'),
            dropdownStride.property('value'), dropdownPadding.property('value'))
    } else if (value == layerNames[1]){
        const layerShow = layerDiv.append('img')
            .attr('width', 90)
            .attr('height', 90)
            .attr('src', '/images/ReLU.png')
        const labelInput = layerDiv.append('label')
            .text('Input: ' + Number(layerDiv.property('id')))
            .attr('class', 'input')
            .style('float', 'left')
        const labelActivation = layerDiv.append('label')
            .text('Activation: ')
            .style('float', 'left')
        const dropdownActivation = layerDiv.append('select')
            .attr('class', 'activationLayer')
        dropdownActivation.selectAll('option').data(activationLayers).enter().append('option')
            .attr('value', d => d)
            .html(d => {console.log(d); return d})
        const labelOutput = layerDiv.append('label')
            .text('Output: ')
            .style('float', 'left')
        const outputTextField = layerDiv.append('text')
            .attr('class', 'outputTextField')
            .text(Number(layerDiv.property('id')))
        dropdownActivation.on('change', function(){
            layerShow.attr('src', '/images/'.concat(d3.select(this).property('value'),'.png'))
        })
    } else if (value == layerNames[2]){
        const layerShow = layerDiv.append('svg')
            .attr('width', 90)
            .attr('height', 90)
        
        const labelInput = layerDiv.append('label')
            .text('Input: ' + Number(layerDiv.property('id')))
            .attr('class', 'input')
            .style('float', 'left')
        const labelFilter = layerDiv.append('label')
            .attr('class', 'filter')
            .text('Filter: ')
            .style('float', 'left')
        const dropdownFilter = layerDiv.append('select')
            .attr('class', 'poolFilter')
        dropdownFilter.selectAll('option').data(poolSizes)
            .enter().append('option')
            .attr('value', d => d)
            .html(d =>{ return d + '*' + d})
        dropdownFilter.on('change', function(){
            f = dropdownFilter.property('value')
            s = f;
            outputSize = Math.round((Number(layerDiv.property('id')) + 2 * p - f) / s) + 1
            outputTextField.text(outputSize)
            labelStride.text('Stride: ' + s)
            layerShow.select('.filterSquare')
                .transition()
                .duration(Duration)
                .attr('width', f * EXPANDSVG)
                .attr('height', f * EXPANDSVG)
            
            layerShow.select('.padSquare')
                .transition()
                .duration(Duration)
                .attr('width', (outputSize + 2 * p) * EXPANDSVG)
                .attr('height', (outputSize + 2 * p) * EXPANDSVG)
            layerShow.select('.outputSquare')
                .transition()
                .duration(Duration)
                .attr('width', (outputSize) * EXPANDSVG)
                .attr('height', (outputSize) * EXPANDSVG)

            const grandParent = this.parentNode.parentNode
            changeNextInputSize(grandParent)            
        })
        const labelStride = layerDiv.append('label')
            .text('Stride: ' + String(d3.select('.poolFilter').property('value')))
            .attr('class', 'poolStride')
            .style('float', 'left')
            .style('clear', 'right')
        const labelPadding = layerDiv.append('label')
            .attr('class', 'pad')
            .text('Padding: ')
            .style('float', 'left')
        const dropdownPadding = layerDiv.append('select')
            .attr('class', 'poolPadding')
            .style('float', 'left')
        dropdownPadding.selectAll('option').data(paddings).enter().append('option')
            .attr('value', d => d)
            .html(d => d)
        dropdownPadding.on('change', function(){
            p = dropdownPadding.property('value')
            outputSize = Math.round((Number(layerDiv.property('id')) + 2 * p - f) / s) + 1
            outputTextField.text(outputSize)
            
            layerShow.select('.padSquare')
                .transition()
                .duration(Duration)
                .attr('width', (outputSize + 2 * p) * EXPANDSVG)
                .attr('height', (outputSize + 2 * p) * EXPANDSVG)
            layerShow.select('.outputSquare')
                .transition()
                .duration(Duration)
                .attr('x', p * EXPANDSVG)
                .attr('y', p * EXPANDSVG)
                .attr('width', outputSize * EXPANDSVG)
                .attr('height', outputSize * EXPANDSVG)
            
            layerShow.select('.filterSquare')
                .transition()
                .duration(Duration)
                .attr('x', p * EXPANDSVG)
                .attr('y', p * EXPANDSVG)

            const grandParent = this.parentNode.parentNode
            changeNextInputSize(grandParent)          
        })
        let f = dropdownFilter.property('value')
        let p = dropdownPadding.property('value')
        let s = dropdownFilter.property('value')
        outputSize = Math.round((Number(layerDiv.property('id')) + 2 * p - f) / s) + 1
        console.log('inputSize: ' + Number(layerDiv.property('id')) + ', f: ' + f + ", p: " + p + ', s: ' + s + ', outputSize:' + outputSize)

        const labelOutput = layerDiv.append('label')
            .text('Output: ')
            .style('float', 'left')
        const outputTextField = layerDiv.append('text')
            .attr('class', 'outputTextField')
            .text(outputSize)
        showSizes(layerShow, Number(layerDiv.property('id')), outputSize, dropdownFilter.property('value'),
            dropdownFilter.property('value'), dropdownPadding.property('value'))
    } else if (value == layerNames[3]){
        const layerShow = layerDiv.append('img')
            .attr('width', 90)
            .attr('height', 90)
            .attr('src', '/images/linear.png')
        
        const labelInput = layerDiv.append('label')
            .text('Input: ' + Number(layerDiv.property('id')))
            .attr('class', 'input')
        const labelOutput = layerDiv.append('label')
            .text('Output: ')
            .style('float', 'left')
        const outputTextField = layerDiv.append('text')
            .attr('class', 'outputTextField')
            .text(10)
    }
    
}

const addBtnHtml = "<i class = 'fas fa-plus-circle fa'></i>"
const closeBtnHtml = "<i class = 'fas fa-window-close fa'></i>"
const addLayerFunc = function () {
    console.log(this.parentNode.parentNode)
    const newLayers = d3.select(this.parentNode.parentNode);
    console.log(this.parentNode.nextElementSibling)
    let newDiv;
    if(this.parentNode.nextElementSibling === null){
        newDiv = newLayers.append('div')
    }
    else{
        newDiv = d3.select(this).select(function(){
            return this.parentNode.parentNode.insertBefore(document.createElement("div"), this.parentNode.nextElementSibling);
        })
    }
    newDiv.attr('class', 'addLayerDiv')
        .attr('id','LayerDiv' + numLayer)
    const newButtons = newDiv.append('div')
        .attr('class', 'addLayerGroup')
        .attr('id', 'Layer' + numLayer)
    var i;
    const closeBtn = newButtons.append('button')
        .attr('class', 'closeBtn')
        .html(closeBtnHtml)
        .on('click', closeFunc)
    for(i = 0; i < 4; i++){
        var newButton = newButtons.append('button')
            .attr('class', 'addLayerBtn')
            .attr('value', layerNames[i])
            .text(layerNames[i])
        newButton.on('mouseover', function(){ d3.select(this).style('background-color', 'green')})
        newButton.on('mouseleave', function(){ d3.select(this).style('background-color', '#4CAF50')})
        newButton.on('click', function(){
            var value = d3.select(this).property('value');
            // makeLayer -> deletion here
            const siblingGPNext = this.parentNode.parentNode.nextElementSibling;
            const siblingGPPrev = this.parentNode.parentNode.previousElementSibling;
            d3.select(this.parentNode.parentNode).remove()
            let newLayer;
            if(this.parentNode.nextElementSibling === null){
                newLayer = newLayers.append('div')
            } else{
                newLayer = newLayers.select(function(){
                    // console.log(this.parentNode)
                    // console.log(this.children)
                    return this
                        .insertBefore(document.createElement("div"), siblingGPNext);
                })
            }
            newLayer.attr('class', 'addLayerDiv newLayer')
            const newNewLayer = newLayer.append('div')
            console.log('siblingGPPrev: ' + siblingGPPrev)
            makeLayer(newNewLayer, siblingGPPrev, value)
            newLayer.append('button')
                .html(addBtnHtml)
                .attr('class', 'addBtn')
                .attr('id', 'addBtn' + numAddBtn)
                .attr('value', numAddBtn)
            var addLayerAgain = d3.selectAll('.addBtn').on('click', addLayerFunc)
            const highlightLayer = d3.selectAll('.addLayerDiv').on('mouseover', function(){
                d3.select(this).style('border', '5px solid pink')
            })
            const deHighlightLayer = d3.selectAll('.addLayerDiv').on('mouseleave', function(){
                d3.select(this).style('border', 'none')
            })
            changeNextInputSize(siblingGPNext)
        })
    }
    let selected = d3.select(this).property('value')
    newDiv.append('button')
        .html(addBtnHtml)
        .attr('class', 'addBtn')
        .attr('id', 'addBtn'+numAddBtn)
        .attr('value', numAddBtn)
    var addLayerAgain = d3.selectAll('.addBtn').on('click', addLayerFunc)
    numLayer++;
    numAddBtn++;
}
var addLayer = d3.selectAll('.addBtn').on('click', addLayerFunc)

const libAreas = ['PytorchArea', 'tensorflowArea', 'kerasArea']

const chooseLib = d3.selectAll('.library').on('click', function(){
    var i, tabcontent, tablinks;
    tabcontent = [];
    for(i = 0; i < libAreas.length; i++){
        Array.prototype.push.apply(tabcontent, document.getElementsByClassName(libAreas[i]));
    }
    console.log(tabcontent);
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    tablinks = document.getElementsByClassName("library");

    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    const currId = d3.select('.MLstep.active').property('value') + d3.select(this).property('id') + 'Area';
    console.log(d3.select('.MLstep.active').property('value') + 
        d3.select(this).property('id') + 'Area')
    document.getElementById(currId).style.display = "block";
    d3.select(this).attr('class', 'library active')


    d3.selectAll('.library').style('border', 'none')
    d3.select(this).style('border-top', '2px solid red')
    currLib = d3.select(this).property('value')
    const codeAreaId = d3.select('.MLstep.active').property('id') + 'Area'
    console.log('codeAreaId: ' + codeAreaId)
    const codeSpace = d3.select('#'.concat(codeAreaId)).select('.' + currLib + 'Area')
    modelLayers = d3.selectAll('.newLayer')
    
    if(currLib == 'Pytorch'){
        const PytorchInit = codeSpace.select('#__init__')
        const PytorchForward = codeSpace.select('#forward')
        PytorchInit.selectAll('p').remove();
        PytorchForward.selectAll('p').remove();
        let numConv = 1, numPool = 1;
        modelLayers.each(function(d, i, nodes){
            const layerDiv = d3.select(this)
            const layerName = layerDiv.select('.layerTitleBtn').node().innerText;
            console.log('layerName: ' + layerName);
            const PytorchInitParagraph = PytorchInit.append('p')
            const PytorchForwardParagraph = PytorchForward.append('p')
            if(layerName == layerNames[0]){ //conv
                const chanIn = layerDiv.select('.chanIn').node().innerText.replace("Ch in: ",'')
                PytorchInitParagraph.append('span')
                    .html('&emsp;&emsp;self.conv' + numConv + ' = nn.Conv2d(')
                const channelInSpan = PytorchInitParagraph.append('span')
                    .attr('title', 'input channel size')
                    .text(chanIn + ', ')
                const channelSize = layerDiv.select('.convChannel').property('value')
                const dropdownChannel = PytorchInitParagraph.append('select')
                    .attr('class', 'convChannelCode')
                    .attr('title', 'output channel size')
                dropdownChannel.selectAll('option').data(channels)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == channelSize; })
                
                PytorchInitParagraph.append('span')
                    .text(', ')

                const filterSize = layerDiv.select('.convFilter').property('value')
                const dropdownFilter = PytorchInitParagraph.append('select')
                    .attr('class', 'convFilterCode')
                    .attr('title', 'kernel(filter) size')
                dropdownFilter.selectAll('option').data(filterSizes)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == filterSize; })
                
                PytorchInitParagraph.append('span')
                    .text(', ')
                const strideSize = layerDiv.select('.convStride').property('value')
                const dropdownStride = PytorchInitParagraph.append('select')
                    .attr('class', 'convStrideCode')
                    .attr('title', 'stride size')
                dropdownStride.selectAll('option').data(strides)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == strideSize; })
                PytorchInitParagraph.append('span')
                    .text(', ')
                const paddingSize = layerDiv.select('.convPadding').property('value')
                const dropdownPadding = PytorchInitParagraph.append('select')
                    .attr('class', 'convPaddingCode')
                    .attr('title', 'padding size')
                dropdownPadding.selectAll('option').data(paddings)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == paddingSize; })
                PytorchInitParagraph.append('span')
                    .text(')')
                //forward
                PytorchForwardParagraph.append('span')
                    .html('&emsp;&emsp;x = self.conv' + numConv + '(x)')
                numConv += 1;
            } else if(layerName == layerNames[1]){
                const activation = layerDiv.select('.activationLayer').property('value')
                PytorchInitParagraph.append('span')
                    .html('&emsp;&emsp;self.' + activation + ' = nn.')
                if(!usedActivationLayers.includes(activation)){
                    const dropdownActivation = PytorchInitParagraph.append('select')
                        .attr('class', 'activationLayerCode')
                        .attr('title', 'Activation function')
                    dropdownActivation.selectAll('option').data(activationLayers)
                        .enter().append('option')
                        .attr('value', d => d)
                        .html(d => d)
                        .property("selected", function(d){ return d == activation; })
                    PytorchInitParagraph.append('span')
                        .text('()')
                    //forward
                    PytorchForwardParagraph.append('span')
                        .html('&emsp;&emsp;x = self.' + activation + '(x)')
                } 
            } else if(layerName == layerNames[2]){
                PytorchInitParagraph.append('span')
                    .html('&emsp;&emsp;self.pool' + numPool + ' = nn.MaxPool2d(')
                const filterSize = layerDiv.select('.poolFilter').property('value')
                const dropdownFilter = PytorchInitParagraph.append('select')
                    .attr('class', 'poolFilterCode')
                    .attr('title', 'kernel(filter) size')
                dropdownFilter.selectAll('option').data(filterSizes)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d === filterSize; })
                
                PytorchInitParagraph.append('span')
                    .text(', ')
                const strideSize = filterSize
                const dropdownStride = PytorchInitParagraph.append('span')
                    .attr('class', 'poolStrideCode')
                    .attr('title', 'stride size(same as kernel size)')
                    .text(strideSize)
                PytorchInitParagraph.append('span')
                    .text(', ')
                const paddingSize = layerDiv.select('.poolPadding').property('value')
                const dropdownPadding = PytorchInitParagraph.append('select')
                    .attr('class', 'poolPaddingCode')
                    .attr('title', 'padding size')
                dropdownPadding.selectAll('option').data(paddings)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d === paddingSize; })
                PytorchInitParagraph.append('span')
                    .text(')')
                
                //forward
                PytorchForwardParagraph.append('span')
                    .html('&emsp;&emsp;x = self.pool' + numPool + '(x)')
                numPool += 1;
            } else if(layerName == layerNames[3]){
                const outputSize = Math.pow(layerDiv.select('div').property('id'), 2) 
                    * channelCheck(layerDiv.node().previousElementSibling);
                PytorchInitParagraph.append('span')
                    .html('&emsp;&emsp;self.fc = ' + ' = nn.Linear(' + outputSize + ', 10)')
                
                //forward
                PytorchForwardParagraph.append('span')
                    .html('&emsp;&emsp;x = x.view(-1, ' + outputSize + ')')
                PytorchForwardParagraph.append('p')
                    .html('&emsp;&emsp;x = self.fc(x)')
            }

        })
    } else if(currLib == 'tensorflow'){
        const tfSpace = d3.select('#modeltensorflowArea')
        tfSpace.selectAll('p').remove();
        modelLayers.each(function(d, i, nodes){
            const layerDiv = d3.select(this)
            const layerName = layerDiv.select('.layerTitleBtn').node().innerText;
            const tensorflowLayer = tfSpace.append('p')
            if(layerName == layerNames[0]){ //conv
                tensorflowLayer.html('x = tf.layers.conv2d(x, ')
                
                const channelSize = layerDiv.select('.convChannel').property('value')
                const dropdownChannel = tensorflowLayer.append('select')
                    .attr('class', 'convChannelCode')
                    .attr('title', 'output channel size')
                dropdownChannel.selectAll('option').data(channels)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == channelSize; })
                
                tensorflowLayer.append('span')
                    .text(', ')

                const filterSize = layerDiv.select('.convFilter').property('value')
                const dropdownFilter = tensorflowLayer.append('select')
                    .attr('class', 'convFilterCode')
                    .attr('title', 'kernel(filter) size')
                dropdownFilter.selectAll('option').data(filterSizes)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == filterSize; })
                
                tensorflowLayer.append('span')
                    .text(', ')
                const strideSize = layerDiv.select('.convStride').property('value')
                const dropdownStride = tensorflowLayer.append('select')
                    .attr('class', 'convStrideCode')
                    .attr('title', 'stride size')
                dropdownStride.selectAll('option').data(strides)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == strideSize; })
                tensorflowLayer.append('span')
                    .text(', ')
                const paddingSize = layerDiv.select('.convPadding').property('value')
                const dropdownPadding = tensorflowLayer.append('select')
                    .attr('class', 'convPaddingCode')
                    .attr('title', 'padding size')
                dropdownPadding.selectAll('option').data(tfPaddings)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d, i){ 
                        return i == paddingSize; 
                    })
                tensorflowLayer.append('span')
                    .text(')')
            } else if(layerName == layerNames[1]){
                const activation = layerDiv.select('.activationLayer').property('value')
                tensorflowLayer.append('span')
                    .html('x = tf.nn.') 
                const dropdownActivation = tensorflowLayer.append('select')
                    .attr('class', 'activationLayerCode')
                    .attr('title', 'Activation function')
                dropdownActivation.selectAll('option').data(activationLayers)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d.toLowerCase())
                    .property("selected", function(d){ return d == activation; })
                tensorflowLayer.append('span')
                    .text('(x)')
                    
            } else if(layerName == layerNames[2]){
                tensorflowLayer.append('span')
                    .html('x = tf.layers.MaxPooling2d(x, ')
                const filterSize = layerDiv.select('.poolFilter').property('value')
                const dropdownFilter = tensorflowLayer.append('select')
                    .attr('class', 'poolFilterCode')
                    .attr('title', 'kernel(filter) size')
                dropdownFilter.selectAll('option').data(filterSizes)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d === filterSize; })
                
                tensorflowLayer.append('span')
                    .text(', ')
                const strideSize = filterSize
                const dropdownStride = tensorflowLayer.append('span')
                    .attr('class', 'poolStrideCode')
                    .attr('title', 'stride size(same as kernel size)')
                    .text(strideSize)
                tensorflowLayer.append('span')
                    .text(', ')
                const paddingSize = layerDiv.select('.poolPadding').property('value')
                const dropdownPadding = tensorflowLayer.append('select')
                    .attr('class', 'poolPaddingCode')
                    .attr('title', 'padding size')
                dropdownPadding.selectAll('option').data(tfPaddings)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d, i){ 
                        return i == paddingSize; 
                    })
                tensorflowLayer.append('span')
                    .text(')')
                
            } else if(layerName == layerNames[3]){
                tensorflowLayer.append('span')
                    .html('x = tf.layers.flatten(x)<br>')
                
                tensorflowLayer.append('span')
                    .html('logits = tf.layers.dense(x, 10)')
            }

        })
    } else if(currLib == 'keras'){
        const kerasSpace = d3.select('#modelkerasArea')
        kerasSpace.selectAll('p').remove();
        modelLayers.each(function(d, i, nodes){
            const layerDiv = d3.select(this)
            const layerName = layerDiv.select('.layerTitleBtn').node().innerText;
            console.log('layerName: ' + layerName);
            const kerasLayer = kerasSpace.append('p')
            if(layerName == layerNames[0]){ //conv
                kerasLayer.append('span').text('model.add(layers.Conv2D(')
                
                const channelSize = layerDiv.select('.convChannel').property('value')
                const dropdownChannel = kerasLayer.append('select')
                    .attr('class', 'convChannelCode')
                    .attr('title', 'output channel size')
                dropdownChannel.selectAll('option').data(channels)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == channelSize; })
                
                kerasLayer.append('span')
                    .text(', ')

                const filterSize = layerDiv.select('.convFilter').property('value')
                const dropdownFilter = kerasLayer.append('select')
                    .attr('class', 'convFilterCode')
                    .attr('title', 'kernel(filter) size')
                dropdownFilter.selectAll('option').data(poolSizes)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == filterSize; })
                
                kerasLayer.append('span')
                    .text(', ')
                const strideSize = layerDiv.select('.convStride').property('value')
                const dropdownStride = kerasLayer.append('select')
                    .attr('class', 'convStrideCode')
                    .attr('title', 'stride size')
                dropdownStride.selectAll('option').data(strides)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == strideSize; })
                kerasLayer.append('span')
                    .text(', ')
                const paddingSize = layerDiv.select('.convPadding').property('value')
                const dropdownPadding = kerasLayer.append('select')
                    .attr('class', 'convPaddingCode')
                    .attr('title', 'padding size')
                dropdownPadding.selectAll('option').data(tfPaddings)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d, i){ 
                        return i == paddingSize; 
                    })
                if(i < 1){
                    kerasLayer.append('span').text(', input_shape = (28, 28, 1)')
                }
                kerasLayer.append('span')
                    .text('))')
            } else if(layerName == layerNames[1]){
                const activation = layerDiv.select('.activationLayer').property('value')
                kerasLayer.append('span')
                    .html('model.add(layers.Activation(activation ="') 
                const dropdownActivation = kerasLayer.append('select')
                    .attr('class', 'activationLayerCode')
                    .attr('title', 'Activation function')
                dropdownActivation.selectAll('option').data(activationLayers)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d == activation; })
                kerasLayer.append('span')
                    .text('")')
                    
            } else if(layerName == layerNames[2]){
                kerasLayer.append('span')
                    .html('model.add(layers.MaxPooling2d(x, ')
                const filterSize = layerDiv.select('.poolFilter').property('value')
                const dropdownFilter = kerasLayer.append('select')
                    .attr('class', 'poolFilterCode')
                    .attr('title', 'kernel(filter) size')
                dropdownFilter.selectAll('option').data(poolSizes)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d){ return d === filterSize; })
                
                kerasLayer.append('span')
                    .text(', ')
                const strideSize = filterSize
                const dropdownStride = kerasLayer.append('span')
                .attr('class', 'poolStrideCode')
                .attr('title', 'stride size(same as kernel size)')
                .text(strideSize)
                kerasLayer.append('span')
                    .text(', ')
                const paddingSize = layerDiv.select('.poolPadding').property('value')
                const dropdownPadding = kerasLayer.append('select')
                    .attr('class', 'poolPaddingCode')
                    .attr('title', 'padding size')
                dropdownPadding.selectAll('option').data(tfPaddings)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d => d)
                    .property("selected", function(d, i){ 
                        return i == paddingSize; 
                    })
                kerasLayer.append('span')
                    .text('))')
                
            } else if(layerName == layerNames[3]){
                kerasLayer.append('span')
                    .html('model.add(layers.Flatten())<br>')
                
                kerasLayer.append('span')
                    .html('model.add(layers.Dense(10))')
            }

        })
    }
})


const mlStepBtns = d3.selectAll('.MLstep')
mlStepBtns.on('click',function () {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("codeArea");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }

    tablinks = document.getElementsByClassName("MLstep");

    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(d3.select(this).property('id').concat('Area')).style.display = "block";
    d3.select(this).attr('class', 'MLstep active')
})








function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
 }





/*********** MACHINE LEARNING USING TENSORFLOW.JS ***********/
async function run() {  
    const data = new MnistData();
    await data.load();
    //   await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
    d3.select('.tf-surface')
        .style('visibility', 'hidden')
        .style('height', '0')
    let oldParent = document.getElementById('tfjs-visor-container');
    let newParent = document.getElementById('resultSpace');
    newParent.appendChild(oldParent);
    await train(model, data);

//   await showAccuracy(model, data);
}  
const EPOCH = 20;
let history = [];
let old_history = [];
  // Model
function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  
    
    let myLayers;
    if(d3.select('#Example').node().innerText == exampleTexts[0]){
        myLayers = d3.select('.addGroup').selectAll('.newLayer')
    } else{
        myLayers = d3.select('.exampleGroup').selectAll('.newLayer')
    }
    myLayers.each(function(d, i ,nodes){
        const layerDiv = d3.select(this)
        const layerName = layerDiv.select('.layerTitleBtn').node().innerText;
        console.log('layerName: ' + layerName);
        const inputSize = Number(layerDiv.select('.input').node().innerText.replace('Input: ', ''))
        const outputSize = Number(layerDiv.select('.outputTextField').node().innerText)
        if(layerName == layerNames[0]){
            let myPadding;
            if(layerDiv.select('.convPadding').property('value') > 0){
                myPadding = 'same';
            }
            else{
                myPadding = 'valid';
            }
            if(i === 0){
                model.add(tf.layers.conv2d({
                    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
                    kernelSize: Number(layerDiv.select('.convFilter').property('value')),
                    filters: Number(layerDiv.select('.chanOut > .convChannel').property('value')),
                    strides: Number(layerDiv.select('.convStride').property('value')),
                    padding: myPadding
                    }));
            } else{
                model.add(tf.layers.conv2d({
                    kernelSize: Number(layerDiv.select('.convFilter').property('value')),
                    filters: Number(layerDiv.select('.chanOut > .convChannel').property('value')),
                    strides: Number(layerDiv.select('.convStride').property('value')),
                    padding: myPadding
                    }));
            }
        } else if(layerName == layerNames[1]){
            const myActivation = layerDiv.select('.activationLayer').property('value').toLowerCase()
            console.log('myActivation: ', myActivation)
            model.add(tf.layers.activation({activation: myActivation}))
        } else if(layerName == layerNames[2]){
            let myPadding;
            if(layerDiv.select('.poolPadding').property('value') > 0){
                myPadding = 'same';
            }
            else{
                myPadding = 'valid';
            }
            if(i === 0){
                model.add(tf.layers.maxPooling2d({
                    poolSize: Number(layerDiv.select('.poolFilter').property('value')), 
                    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
                    padding: myPadding
                }));
            } else{
                model.add(tf.layers.maxPooling2d({
                    poolSize: Number(layerDiv.select('.poolFilter').property('value')), 
                    padding: myPadding
                }));
            }
        } else if(layerName == layerNames[3]){
            model.add(tf.layers.flatten());
            model.add(tf.layers.dense({
                units: NUM_CLASSES
            }));
        }
    })

    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.momentum(Number(d3.select('#learningRate').property('value')),
        Number(d3.select('#momentum').property('value')));
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    
    return model;
    }

    // Training
    async function train(model, data) {
    const metrics = ['acc'];
    const container = {
        name: 'Train Result'
    };
    
    let BATCH_SIZE = Number(d3.select('#batchSize').property('value'));
    console.log('batch size: ', BATCH_SIZE);
    let TRAIN_DATA_SIZE = NUM_TRAIN_ELEMENTS / NUM_CLASSES; // numtraindata / numclasses
    let TEST_DATA_SIZE = NUM_TEST_ELEMENTS / NUM_CLASSES;  // numtestdata / numclasses
    console.log('train data size: ' + TRAIN_DATA_SIZE + ', test data size: ' + TEST_DATA_SIZE)
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
        ];
    });
    
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
        ];
    });
        return model.fit(trainXs, trainYs, {
            batchSize: BATCH_SIZE,
            validationData: [testXs, testYs],
            epochs: EPOCH,
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, log) => {
                    history.push(log);
                    tfvis.show.history(container, history, metrics);
                    if(stopLearning){
                        this.stopTraining = true;
                    }
                    while(pauseLearning){
                        await sleep(100);
                    }
                    d3.select('#epochCurr')
                        .transition()
                        .duration(Duration)
                        .text(++epochCurr)
                    if(epoch + 1 === EPOCH){
                        d3.select('#play').html('<i class="fas fa-play-circle fa-3x"></i>')
                        d3.select('.modelSpace').selectAll('button').property('disabled', true)
                        d3.select('.modelSpace').selectAll('select').property('disabled', true)
                        d3.select('.betweenSpaces').selectAll('select').property('disabled', true)
                        d3.select('.codeSpace').selectAll('select').property('disabled', true)
                        stopLearning = false;
                        pauseLearning = false;
                        old_history = history;
                    }
                }
            }
        });
    }
  
    const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
  
  function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);
  
    testxs.dispose();
    return [preds, labels];
  }
  async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = {name: 'Accuracy', tab: 'Evaluation'};
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
  
    labels.dispose();
  }
  