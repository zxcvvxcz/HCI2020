import {MnistData} from './data.js';



function translate(x, y) {
    return `translate(${x}, ${y})`
}

function showSizes(svg, inp, outp, filt, str, padd){
    console.log('input: ' + inp + 'output: ' + outp + 'filter:' + filt + 'stride: ' + str + 'padd: ' + padd)
    const entireSquare = svg.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 56)
        .attr('height', 56)
        .style('border', '1px solid black')
        .style('fill', 'white')
    const inputSquare = svg.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        // .attr('width', inp * 2)
        // .attr('height', inp * 2)
        .attr('width', 56)
        .attr('height', 56)
        .style('border', '1px solid black')
        .style('fill', 'gray')
        
    const padSquare = svg.append('rect')
    .attr('x', 0)
    .attr('y', 0)
    // .attr('width', outp * 2)
    // .attr('height', outp * 2)
    .attr('width', 20)
    .attr('height', 20)
    .style('border', '1px solid black')
    .style('fill', 'blue')
    const outputSquare = svg.append('rect')
        .attr('x', padd)
        .attr('y', padd)
        .attr('width', (outp - 2 * padd) * 2)
        .attr('height', (outp - 2 * padd) * 2)
        .style('border', '1px solid black')
        .style('fill', 'black')
    
    const filterSquare = svg.append('rect')
        .attr('x', padd)
        .attr('y', padd)
        .attr('width', filt * 2)
        .attr('height', filt * 2)
        .style('border', '1px solid black')
        .style('fill', 'red')
    return;
}
const [btnSvgWidth, btnSvgHeight] = [300, 100];
const [iconWidth, iconHeight] = [50, 50];
const playClick = function () {
    
    if(this.innerHTML == '<i class="fas fa-play-circle fa-3x"></i>'){   //play
        this.innerHTML = '<i class="fas fa-pause-circle fa-3x"></i>';
    }
    else if(this.innerHTML == '<i class="fas fa-pause-circle fa-3x"></i>'){  //pause
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
            .text('Test data:    ' + (65000 - selectedOption))
})
let modelLayers = [];
let layerNames = ['Conv', 'Acti','Pool', 'Linear'];
const activationLayers = ['ReLU', 'tanh', 'sigmoid'];
const filterSizes = [3, 5];
const strides = [1, 2];
const paddings = [0, 1];
const learningRates = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1];

let numLayer = 0;
let numAddBtn = 1;
let layers = [];
let addBtns = [0];
let inputSizes = [28] * 10;
let outputSizes = [10] * 10;

let currLib = 'Pytorch'
let PytorchInit = [];
let PytorchForward = [];
let PytorchCodes = [PytorchInit, PytorchForward];
let tensorflowCodes = [];
let kerasCodes = [];
let modelCodes = [PytorchCodes, tensorflowCodes, kerasCodes];
// const setLearningRate = d3.select('.learningRate').on('change')
const setLayer = (layer, i, inp, outp, spliceNum) => {
    if (layer == layerNames[0]){
        modelLayers.splice(i, spliceNum, {
            layer : layerNames[0],
            input : inp,
            filterSize : filterSizes[0],
            stride : strides[0],
            padding : paddings[0],
            channelIn: 1,
            channelOut: 1,
            output : outp, 
        })
    } else if (layer == layerNames[1]){
        modelLayers.splice(i, spliceNum, {
            layer : layerNames[1],
            method : activationLayers[0], //ReLU
        })
    } else if (layer == layerNames[2]){
        modelLayers.splice(i, spliceNum, {
            layer : layerNames[2],
            filterSize : filterSizes[0],
            stride : strides[0],
            padding : paddings[0],
            output : outp,
        })
    } else if (layer == layerNames[3]){
        modelLayers.splice(i, spliceNum, {
            layer : layerNames[3],
            input : inp,
            output : outp,
        })
    }
}
const closeFunc = function(){
    d3.select(this.parentNode.parentNode).remove()
}
const makeLayer = function (layerDiv, i, value) {
    console.log('Make layer')
    layerDiv.attr('class', 'layerVis'.concat(' ', value))
    // const i = d3.select(layerDiv.parentNode.selectAll('.addLayerBtn')).each(function(d, i){
    //     if(d == layerDiv) return i;
    // })
    console.log('i:' + i + ' value: ' + value)
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
    if(value == layerNames[0]){
        const layerShow = layerDiv.append('svg')
            .attr('width', 90)
            .attr('height', 90)
        const labelInput = layerDiv.append('label')
            .text('Input: ' + inputSizes[i])
            .attr('for', '#convInput' + i)
        const labelFilter = layerDiv.append('label')
            .text('Filter: ')
            .attr('for', '#convFilter' + i)
            .style('float', 'left')
        const dropdownFilter = layerDiv.append('select')
            .attr('id', 'convFilter' + i)
        d3.select('#convFilter' + i).selectAll('option').data(filterSizes)
            .enter().append('option')
            .attr('value', d => d)
            .html(d =>{ return d + '*' + d})
        const labelStride = layerDiv.append('label')
            .text('Stride: ')
            .attr('for', '#convStride' + i)
            .style('float', 'left')
        const dropdownStride = layerDiv.append('select')
            .attr('id', 'convStride' + i)
        d3.select('#convStride' + i).selectAll('option').data(strides).enter().append('option')
            .attr('value', d => d)
            .html(d => d)

        const labelPadding = layerDiv.append('label')
            .text('Padding: ')
            .attr('for', '#convPadding' + i)
            .style('float', 'left')
        const dropdownPadding = layerDiv.append('select')
            .attr('id', 'convPadding' + i)
        d3.select('#convPadding' + i).selectAll('option').data(paddings).enter().append('option')
            .attr('value', d => d)
            .html(d => d)
        const labelChanIn = layerDiv.append('label')
            .text('Channel in: 1')
            .style('float', 'left')
        const labelChanOut = layerDiv.append('label')
            .text('Channel out: 1')
            .style('float', 'left')
        const labelOutput = layerDiv.append('label')
            .text('Output: ' + outputSizes[i])
            .attr('for', '#convOutput' + i)
            .style('float', 'left')
        console.log('i : ' + i)
        showSizes(layerShow, inputSizes[i], outputSizes[i], dropdownFilter.property('value'),
            dropdownStride.property('value'), dropdownPadding.property('value'))
    } else if (value == layerNames[1]){
        const layerShow = layerDiv.append('img')
            .attr('width', 90)
            .attr('height', 90)
            .attr('src', '/images/ReLU.png')
        
        const labelActivation = layerDiv.append('label')
            .text('Activation: ')
            .attr('for', '#activationLayer' + i)
            .style('float', 'left')
        const dropdownActivation = layerDiv.append('select')
            .attr('id', 'activationLayer' + i)
        d3.select('#activationLayer' + i).selectAll('option').data(activationLayers).enter().append('option')
            .attr('value', d => {console.log(d); return d})
            .html(d => d)
        dropdownActivation.on('change', function(){
            layerShow.attr('src', '/images/'.concat(d3.select(this).property('value'),'.png'))
        })
    } else if (value == layerNames[2]){
        const layerShow = layerDiv.append('svg')
            .attr('width', 90)
            .attr('height', 90)
        
        const labelInput = layerDiv.append('label')
            .text('Input: ' + inputSizes[i])
            .attr('for', '#convInput' + i)
        const labelFilter = layerDiv.append('label')
            .text('Filter: ')
            .attr('for', '#convFilter' + i)
            .style('float', 'left')
        const dropdownFilter = layerDiv.append('select')
            .attr('id', 'convFilter' + i)
        d3.select('#convFilter' + i).selectAll('option').data(filterSizes)
            .enter().append('option')
            .attr('value', d => d)
            .html(d =>{ return d + '*' + d})
        const labelStride = layerDiv.append('label')
            .text('Stride: ')
            .attr('for', '#convStride' + i)
            .style('float', 'left')
        const dropdownStride = layerDiv.append('select')
            .attr('id', 'convStride' + i)
        d3.select('#convStride' + i).selectAll('option').data(strides).enter().append('option')
            .attr('value', d => d)
            .html(d => d)
        const labelPadding = layerDiv.append('label')
            .text('Padding: ')
            .attr('for', '#convPadding' + i)
            .style('float', 'left')
        const dropdownPadding = layerDiv.append('select')
            .attr('id', 'convPadding' + i)
        d3.select('#convPadding' + i).selectAll('option').data(paddings).enter().append('option')
            .attr('value', d => d)
            .html(d => d)
        const labelOutput = layerDiv.append('label')
            .text('Output: ' + outputSizes[i])
            .attr('for', '#convOutput' + i)
            .style('float', 'left')
        
        showSizes(layerShow, inputSizes[i], outputSizes[i], dropdownFilter.property('value'),
            dropdownStride.property('value'), dropdownPadding.property('value'))
    } else if (value == layerNames[3]){
        const layerShow = layerDiv.append('img')
            .attr('width', 90)
            .attr('height', 90)
            .attr('src', '/images/linear.png')
        
        const labelInput = layerDiv.append('label')
            .text('Input: ' + inputSizes[i])
            .attr('for', '#linearInput' + i)
        const labelOutput = layerDiv.append('label')
            .text('Output: 10')
            .attr('for', '#linearOutput' + i)
            .style('float', 'left')
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
            i = layers.indexOf(d3.select(this).property('id'));
            var value = d3.select(this).property('value');
            setLayer(value, i, inputSizes[i], outputSizes[i], 0)
            // makeLayer -> deletion here
            const siblingGPNode = this.parentNode.parentNode.nextElementSibling;
            d3.select(this.parentNode.parentNode).remove()
            let newLayer;
            console.log('sib: '+this.parentNode.nextElementSibling)
            console.log('grandparent: '+this.parentNode.parentNode)
            if(this.parentNode.nextElementSibling === null){
                newLayer = newLayers.append('div')
            } else{
                newLayer = newLayers.select(function(){
                    console.log(this.parentNode)
                    console.log(this.children)
                    return this
                        .insertBefore(document.createElement("div"), siblingGPNode);
                })
            }
            newLayer.attr('class', 'addLayerDiv newLayer')
            const newNewLayer = newLayer.append('div')
            makeLayer(newNewLayer, i, value)
            // makeLayer(newNewLayer, value)
            newLayer.append('button')
                .html(addBtnHtml)
                .attr('class', 'addBtn')
                .attr('id', 'addBtn'+numAddBtn)
                .attr('value', numAddBtn)
        var addLayerAgain = d3.selectAll('.addBtn').on('click', addLayerFunc)
        const highlightLayer = d3.selectAll('.addLayerDiv').on('mouseover', function(){
            d3.select(this).style('border', '5px solid pink')
        })
        const deHighlightLayer = d3.selectAll('.addLayerDiv').on('mouseleave', function(){
            d3.select(this).style('border', 'none')
        })
        console.log(modelLayers)
        })
        //
    }
    let selected = d3.select(this).property('value')
    console.log('selected: ' +selected)
    layers.splice(addBtns.indexOf(selected), 0, newButtons.property('id'))
    console.log(layers)
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
    console.log(d3.select('.MLstep .active')) 
    const step = d3.select('.MLstep .active').property('value');
    for(i = 0; i < 3; i++){
        d3.selectAll('.' + libAreas[i]).style('display', 'none')
        d3.select('#' + step + libAreas[i])
            .attr('class', libAreas[i] + 'active')
        d3.selectAll('.' + libAreas[i]).select('.active').style('display', 'block')
    }
    d3.selectAll('.library').style('border', 'none')
    d3.select(this).style('border-top', '2px solid red')
    console.log('#'.concat(d3.selectAll('.MLstep.active').property('id'), 'Area'))
    d3.selectAll('#'.concat(d3.selectAll('.MLstep.active').property('id'), 'Area')).remove()
    currLib = d3.select(this).property('value')
    // const codeSpace = d3.select('.codeSpace').append('div')
    //     .attr('class', 'codeArea')
    //     .attr('id', ''.concat(d3.selectAll('.MLstep.active').property('id'), 'Area'))
    const codeSpace = d3.select('.codeSpace').select('#' + step + currLib + 'Area')
    if(step == 'model'){
        if(currLib == 'Pytorch'){
            codeSpace.append('p').append('text')
                .text('import torch.nn as nn')
            codeSpace.append('p').append('text')
                .html('class Net(nn.Module):')
            const PytorchInit =  codeSpace.append('p')
                .attr('class', 'PytorchInit')
            PytorchInit.append('p').append('text')
                .html('&emsp;def __init__(self):')
            PytorchInit.append('text')
                .html('&emsp;&emsp;super(Net, self).__init__()')
            
            const PytorchForward = codeSpace.append('p')
                .attr('class', 'PytorchForward')
            PytorchForward.append('text')
                .html('&emsp;def forward(self, x):')

            modelLayers.forEach(function(e, i){
                console.log(e);
                const layerName = modelLayers.layer;
                console.log(layerNames[0]);
                if(layerName == layerNames[0]){
                    console.log(e);
                    PytorchInit.append('p')
                        .append('span')
                        .html(String.concat('&emsp;&emsp;self.conv', i, 'nn.Conv2d( ', inputsizes[i], outputSizes[i], ) )
                    PytorchInit.append('select')
                        .attr('id', String.concat('self.conv ', i))
                    d3.select(String.concat('#self.conv ', i)).selectAll('option').data(filterSizes)
                        .enter().append('option')
                        .attr('value', d => d)
                        .html(d =>{ return d + '*' + d})
                } else if(layerName == layerNames[1]){

                } else if(layerName == layerNames[2]){

                } else if(layerName == layerNames[3]){
                    
                }

            })
            codeSpace.append('text').text('net = Net()')
        } else if(currLib == 'tensorflow'){

        } else if(currLib == 'keras'){

        }
    } else if (step == 'preprocess'){

    } else if (step == 'result'){

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
    // evt.currentTarget.className += " active";
})



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

