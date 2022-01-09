class loadModel{
    constructor(){
        this.model;
    }
    load(){
        const MODEL_URL = 'http://127.0.0.1:8887/jsModel/model.json';
        const model = tf.loadLayersModel(MODEL_URL);
        return model
    }

    async prepareModel(){
        this.model = await this.load(); 
        console.log(this.model.summary())
        alert('Model is Loaded')
    }

    async predict(frames) {          
        
        const predictedClass = tf.tidy(() => {
            const pred = this.model.predict(tf.stack(rescaledFrames).expandDims(0));
            return pred.as1D().argMax()
        });

        
        return (await predictedClass.data())[0]
        //   document.getElementById("prediction").innerText = predictionText;
      }
}
