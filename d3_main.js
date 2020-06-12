
export let layerInfo;   //layer information for machine learning


let modelLayers = [];
let layerNames = ['Conv', 'Acti','Pool', 'Linear'];
const activationLayers = ['ReLU', 'tanh', 'sigmoid'];
const filterSizes = [3, 5];
const strides = [1, 2];
const paddings = [0, 1];
const learningRates = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1];
const poolFilterSizes = [2, 3, 4];
let numLayer = 0;
let numAddBtn = 1;
// let layers = [];
let addBtns = [0];


function translate(x, y) {
    return `translate(${x}, ${y})`
}
const EXPANDSVG = 3;
const Duration = 500;
function changeInputSize(layerDiv){
    const layerCategory = layerDiv.select('.layerTitleBtn').node().innerText
    const inputSize = Number(layerDiv.property('id'))
    layerDiv.select('.input').text('Input: ' + inputSize )
    console.log('category: ' + layerCategory)
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
        layerDiv.select('.outputTextField').text(outputSize)
    } else if(layerCategory == layerNames[2]){
        let f = layerDiv.select('.poolFilter').property('value')
        let p = layerDiv.select('.poolPadding').property('value')
        let s = f
        const outputSize = Math.round((inputSize + 2 * p - f) / s) + 1

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
    if(layerDiv.node().parentNode.nextElementSibling !== null
        && layerDiv.node().parentNode.nextElementSibling.className == 'addLayerDiv newLayer'){
        const nextDiv = d3.select(layerDiv.node().parentNode.nextElementSibling).select('div')
        nextDiv.attr('id', outputSize)
        changeInputSize(nextDiv)
    }
}
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
const [btnSvgWidth, btnSvgHeight] = [300, 100];
const [iconWidth, iconHeight] = [50, 50];
const playClick = function () {
    
    if(this.innerHTML == '<i class="fas fa-play-circle fa-3x"></i>'){   //play
        this.innerHTML = '<i class="fas fa-pause-circle fa-3x"></i>';
        //make JSON to be exported
    }
    else if(this.innerHTML == '<i class="fas fa-pause-circle fa-3x"></i>'){  //pause
        this.innerHTML = '<i class="fas fa-play-circle fa-3x"></i>';
        //pause learning
    }
    else{   //stop
        //stop learning
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

let currLib = 'Pytorch'
// let PytorchInit = [];
// let PytorchForward = [];
// let PytorchCodes = [PytorchInit, PytorchForward];
// let tensorflowCodes = [];
// let kerasCodes = [];
// let modelCodes = [PytorchCodes, tensorflowCodes, kerasCodes];
// const setLearningRate = d3.select('.learningRate').on('change')
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
    if(grandParent.nextElementSibling !== null){
        const layerDiv = d3.select(grandParent).select('div')
        const nextDiv = d3.select(grandParent.nextElementSibling).select('div')
        if(prevGP !== null){
            if(prevGP.className == 'addLayerDiv newLayer'){
                nextDiv.attr('id', d3.select(prevGP).select('.outputTextField').node().innerText)
                console.log('layerDiv id: ' + layerDiv.property('id'))
                nextDiv.select('.input')
                .text('Input: ' + Number(layerDiv.property('id')))
            }
        }
        else{
            nextDiv.attr('id', 28)
        } 
        changeInputSize(nextDiv)
    }
}
const closeFunc = function(){
    const grandParent = this.parentNode.parentNode;
    const prevGP = grandParent.previousElementSibling;
    changeNextInputSize(grandParent);
    // if(grandParent.nextElementSibling !== null
    //     && grandParent.nextElementSibling.className == 'addLayerDiv newLayer'){
    //     const nextDiv = d3.select(grandParent.nextElementSibling).select('div')
    //     while(prevGP !== null){
    //         if(prevGP.className == 'addLayerDiv newLayer'){
    //             nextDiv.attr('id', d3.select(prevGP).select('.outputTextField').node().innerText)
    //             console.log('layerDiv id: ' + layerDiv.property('id'))
    //             nextDiv.select('.input')
    //             .text('Input: ' + Number(layerDiv.property('id')))
    //             break;
    //         }
    //     }
    //     if(prevGP === null){
    //         nextDiv.attr('id', 28)
    //     } 
    //     changeInputSize(nextDiv)
    // }
    d3.select(this.parentNode.parentNode).remove()
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
    let inputSize = 28
    let outputSize;
    if(prevSibling.classList.contains('newLayer')){
        inputSize = Number(d3.select(prevSibling).select('.layerVis').select('.outputTextField').node().innerText);
    }
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
            changeNextInputSize(grandParent)
            // if(this.parentNode.parentNode.nextElementSibling !== null
            //     && this.parentNode.parentNode.nextElementSibling.className == 'addLayerDiv newLayer'){
            //         const nextDiv = d3.select(grandParent.nextElementSibling).select('div')
            //     while(prevGP !== null){
            //         if(prevGP.className == 'addLayerDiv newLayer'){
            //             nextDiv.attr('id', d3.select(prevGP).select('.outputTextField').node().innerText)
            //             console.log('layerDiv id: ' + layerDiv.property('id'))
            //             nextDiv.select('.input')
            //             .text('Input: ' + Number(layerDiv.property('id')))
            //             break;
            //         }
            //     }
            //     if(prevGP === null){
            //         nextDiv.attr('id', 28)
            //     } 
            //     // changeInputSize(nextDiv)
            //     // const nextDiv = d3.select(this.parentNode.parentNode.nextElementSibling).select('div')
            //     nextDiv.attr('id', outputSize)
            //     changeInputSize(nextDiv)
            // }
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
            changeNextInputSize(grandParent)
            // if(this.parentNode.parentNode.nextElementSibling !== null
            //     && this.parentNode.parentNode.nextElementSibling.className == 'addLayerDiv newLayer'){
            //     const nextDiv = d3.select(this.parentNode.parentNode.nextElementSibling).select('div')
            //     nextDiv.attr('id', outputSize)
            //     changeInputSize(nextDiv)
            // }
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
            
            const grandParent = this.parentNode.parentNode
            changeNextInputSize(grandParent);
            // if(this.parentNode.parentNode.nextElementSibling != null 
            //     && this.parentNode.parentNode.nextElementSibling.className == 'addLayerDiv newLayer'){
            //     const nextDiv = d3.select(this.parentNode.parentNode.nextElementSibling).select('div')
            //     nextDiv.attr('id', outputSize)
            //     changeInputSize(nextDiv)
            // }
        })
        const labelChanIn = layerDiv.append('label')
            .text('Channel in: 1')
            .style('float', 'left')
        const labelChanOut = layerDiv.append('label')
            .text('Channel out: 1')
            .style('float', 'left')

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
            .text('Filter: ')
            .style('float', 'left')
        const dropdownFilter = layerDiv.append('select')
            .attr('class', 'poolFilter')
        dropdownFilter.selectAll('option').data(poolFilterSizes)
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
            changeNextInputSize(grandParent)            // if(this.parentNode.parentNode.nextElementSibling !== null
            //     && this.parentNode.parentNode.nextElementSibling.className == 'addLayerDiv newLayer'){
            //     const nextDiv = d3.select(this.parentNode.parentNode.nextElementSibling).select('div')
            //     nextDiv.attr('id', outputSize)
            //     changeInputSize(nextDiv)
            // }
        })
        const labelStride = layerDiv.append('label')
            .text('Stride: ' + String(d3.select('.poolFilter').property('value')))
            .style('float', 'left')
            .style('clear', 'right')
        const labelPadding = layerDiv.append('label')
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
            changeNextInputSize(grandParent)            // if(this.parentNode.parentNode.nextElementSibling !== null 
            //     && this.parentNode.parentNode.nextElementSibling.className == 'addLayerDiv newLayer'){
            //     const nextDiv = d3.select(this.parentNode.parentNode.nextElementSibling).select('div')
            //     nextDiv.attr('id', outputSize)
            //     changeInputSize(nextDiv)
            // }
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
            // makeLayer(newNewLayer, i, value)
            console.log(siblingGPPrev)
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
            // if(siblingGPNext !== null && siblingGPNext.className == 'addLayerDiv newLayer'){
            //     const nextDiv = d3.select(siblingGPNext).select('div')
            //     nextDiv.attr('id', outputSize)
            //     changeInputSize(nextDiv)
            // }
        })
    }
    let selected = d3.select(this).property('value')
    console.log('selected: ' +selected)
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
    // console.log('#'.concat(d3.selectAll('.MLstep.active').property('id'), 'Area'))
    // d3.selectAll('#'.concat(currId)).remove()
    currLib = d3.select(this).property('value')
    const codeAreaId = d3.select('.MLstep.active').property('id') + 'Area'
    console.log('codeAreaId: ' + codeAreaId)
    const codeSpace = d3.select('#'.concat(codeAreaId)).select('.' + currLib + 'Area')
        // .attr('class', currLib + 'Area')
        // .attr('id', currId)
    if(currLib == 'Pytorch'){
        // codeSpace.append('p').append('text')
        //     .text('import torch.nn as nn')
        // codeSpace.append('p').append('text')
        //     .html('class Net(nn.Module):')
        // const PytorchInit =  codeSpace.append('p')
        //     .attr('class', 'PytorchInit')
        // PytorchInit.append('p').append('text')
        //     .html('&emsp;def __init__(self):')
        // PytorchInit.append('text')
        //     .html('&emsp;&emsp;super(Net, self).__init__()')
        const PytorchInit = codeSpace.select('#__init__')
        const PytorchForward = codeSpace.select('#forward')
        PytorchForward.append('text')
            .html('&emsp;def forward(self, x):')
        modelLayers = d3.selectAll('.newLayer')
        let numConv = 1, numPool = 1;
        let activations = [];
        modelLayers.each(function(d, i, nodes){
            const layerName = d3.select(this).select('.layerTitleBtn').node().innerText;
            const inputSize = Number(d3.select(this).select('.input').node().innerText.replace('Input: ', ''))
            const outputSize = Number(d3.select(this).select('.outputTextField').node().innerText)
            if(layerName == layerNames[0]){ //conv
                PytorchInit.append('p')
                    .append('span')
                    .html(String.concat('&emsp;&emsp;self.conv', numConv, 'nn.Conv2d( ', inputsizes[i], outputSizes[i]) )
                PytorchInit.append('select')
                    .attr('id', String.concat('self.conv ', numConv))
                d3.select(String.concat('#self.conv ', numConv)).selectAll('option').data(filterSizes)
                    .enter().append('option')
                    .attr('value', d => d)
                    .html(d =>{ return d + '*' + d})
                numConv += 1;
            } else if(layerName == layerNames[1]){

            } else if(layerName == layerNames[2]){

            } else if(layerName == layerNames[3]){
                
            }

        })
        codeSpace.append('text').text('net = Net()')
    } else if(currLib == 'tensorflow'){

    } else if(currLib == 'keras'){

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
