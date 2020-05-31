import {MnistData} from './data.js';

function translate(x, y) {
    return `translate(${x}, ${y})`
}
const [btnSvgWidth, btnSvgHeight] = [300, 100];
const [iconWidth, iconHeight] = [50, 50];
const playClick = function () {
    
    if(this.innerHTML === '<i class="fas fa-play-circle fa-3x"></i>'){   //play
        this.innerHTML = '<i class="fas fa-pause-circle fa-3x"></i>';
    }
    else if(this.innerHTML === '<i class="fas fa-pause-circle fa-3x"></i>'){  //pause
        this.innerHTML = '<i class="fas fa-play-circle fa-3x"></i>';
    }
    else{   //stop

    }
}
const btnTopSvg = d3.selectAll('.btnTop')
const btnLearnFunc = btnTopSvg.selectAll('.btnLearning').on('click', playClick);

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
            .text('Test data:    ' + (65000 - selectedOption) + '    ')
})

let modelLayers = [];
const activationLayers = ['ReLU', 'tanh', 'sigmoid'];
const filterSizes = [3, 5];
const strides = [1, 2];
const paddings = [0, 1];
const learningRates = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1];
// const setLearningRate = d3.select('.learningRate').on('change')
const setLayer = (layer, i, inp, outp, spliceNum) => {
    if (layer === 'conv'){
        modelLayers.splice(i, spliceNum, {
            layer : 'conv',
            input : inp,
            filterSize : filterSizes[0],
            stride : strides[0],
            padding : paddings[0],
            channelIn: 1,
            channelOut: 1,
            output : outp, 
        })
    } else if (layer === 'activation'){
        modelLayers.splice(i, spliceNum, {
            layer : 'activation',
            method : activationLayers[0], //ReLU
        })
    } else if (layer === 'pool'){
        modelLayers.splice(i, spliceNum, {
            layer : 'pool',
            filterSize : filterSizes[0],
            stride : strides[0],
            padding : paddings[0],
            output : outp,
        })
    } else if (layer === 'linear'){
        modelLayers.splice(i, spliceNum, {
            layer : 'linear',
            input : inp,
            output : outp,
        })
    }
}

const updateLayer = () => {

}
async function showExamples(data) {
    // Create a container in the visor
    const surface = dataSvg

    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

