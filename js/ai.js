function AI(grid) {
  this.grid = grid;
  this.input = [[0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]]
  this.model = new KerasJS.Model({
    filepath: 'models/model_flat.bin',
    gpu: true
  })
}

AI.prototype.getBoard = function() {
  var board = []
  for(i in [0,1,2,3]){
    for(j in [0,1,2,3]){
      if(this.grid.cells[j][i] == null){
        board.push(0);
      } else {
        board.push(Math.log2(this.grid.cells[j][i].value));
      }
    }
  }
  return board
}


AI.prototype.predict = function() {
  inputBoard = this.getBoard()
  return this.model
    .ready()
    .then(() => {
      // input data object keyed by names of the input layers
      // or `input` for Sequential models
      // values are the flattened Float32Array data
      // (input tensor shapes are specified in the model config)
      const inputData = {
        input_1: new Float32Array(inputBoard)
      }

      // make predictions
      return this.model.predict(inputData)
    })
    .then(outputData => {
      // outputData is an object keyed by names of the output layers
      // or `output` for Sequential models
      // e.g.,
      // outputData['fc1000']

      var move = this.maxIndex(outputData['policy_out'])
      var translated = this.changeEncoding(move)
      return translated
    })
    .catch(err => {
      console.log(err)
    })
}

AI.prototype.getBest = function() {
  return this.predict().then( moveAnswer => {
    return {move : moveAnswer};
  });
}

AI.prototype.maxIndex = function(array) {
  var bestIndex = 0;
  var bestValue = 0;
  for (i in [0,1,2,3]) {
      if (array[i] > bestValue) {
        bestValue = array[i];
        bestIndex = i;
      }
  }
  return bestIndex;
}

AI.prototype.changeEncoding = function(move) {
  return {
    0:0,
    1:2,
    2:3,
    3:1
  }[move];
}


AI.prototype.translate = function(move) {
 return {
    0: 'up',
    1: 'right',
    2: 'down',
    3: 'left'
  }[move];
}
