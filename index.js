const {spawn} = require("child_process");

// const inputData = [58,0,0,100,248,0,0,122,0,1,1,0,2];

// const inputData = [62,0,0,138,294,1,1,106,0,1.9,1,3,2]
const formData = {
    age: 62,
    sex: 0,
    cp: 0,
    trestbps: 138,
    chol: 294,
    fbs: 1,
    restecg: 1,
    thalach: 106,
    exang: 0,
    oldpeak: 1.9,
    slope: 1,
    ca: 3,
    thal: 2
};

const inputData = Object.values(formData);


const pythonProcess = spawn('python', ["predict.py"]);

// Send input data to the Python script
pythonProcess.stdin.write(JSON.stringify(inputData));
pythonProcess.stdin.end();

pythonProcess.stdout.on('data', (data) => {
    console.log(data.toString());
});

pythonProcess.stderr.on('data', (data) => {
    console.error(data.toString());
});

pythonProcess.on('exit', (code) => {
    console.log(`PythonProcess exited with code ${code}`);
});