const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const resultText = document.getElementById('result');
const statusText = document.getElementById('statusText');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const uploadBtn = document.getElementById('customUploadBtn');

let mySession = null; 

// 网页一打开，手动拦截并下载模型
async function initModel() {
    try {
        progressContainer.style.display = 'block';
        statusText.innerText = "开始制作尺子";

        // 1. 发起请求，但不直接拿结果，而是拿到一个“水管”(reader)
        const response = await fetch('./my_aesthetic_scorer_web.onnx');
        const contentLength = response.headers.get('content-length'); // 获取文件总大小
        const totalBytes = parseInt(contentLength, 10);
        let loadedBytes = 0;

        const reader = response.body.getReader();
        const chunks = []; // 用来存每一滴水

        // 2. 循环接水，直到接满
        while (true) {
            const { done, value } = await reader.read();
            if (done) break; // 下载完了
            
            chunks.push(value);
            loadedBytes += value.length;

            // 实时计算进度并更新进度条！
            if (totalBytes) {
                let percent = Math.round((loadedBytes / totalBytes) * 100);
                progressBar.style.width = percent + '%';
                statusText.innerText = `⏳ 加载中: ${percent}% (${(loadedBytes/1024/1024).toFixed(1)} MB / ${(totalBytes/1024/1024).toFixed(1)} MB)`;
            } else {
                statusText.innerText = `⏳ 加载中: ${(loadedBytes/1024/1024).toFixed(1)} MB...`;
            }
        }

        statusText.innerText = "尺子制作完成！正在初始化...";

        // 3. 把所有的水滴拼成一个完整的文件
        const modelBuffer = new Uint8Array(loadedBytes);
        let position = 0;
        for (let chunk of chunks) {
            modelBuffer.set(chunk, position);
            position += chunk.length;
        }

        // 4. 把拼好的文件直接喂给 ONNX Runtime！
        mySession = await ort.InferenceSession.create(modelBuffer);
        
        // 5. 准备就绪，隐藏进度条，点亮按钮
        progressContainer.style.display = 'none';
        statusText.style.display = 'none';
        uploadBtn.innerText = "📸 点击上传照片测量";
        uploadBtn.disabled = false; 
        resultText.innerText = "尺子准备就绪！";
        resultText.style.color = "#4CAF50";

    } catch (e) {
        console.error(e);
        statusText.innerText = "❌ 加载失败，请检查网络后刷新网页。";
        statusText.style.color = "red";
    }
}

// 启动！
initModel();

// 后面的测图逻辑和之前一样（假进度条处理计算瞬间）
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file || !mySession) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        preview.src = event.target.result;
        preview.style.display = 'inline-block';
        
        // 计算时显示一个伪进度效果
        resultText.innerText = "照片已收到，正在测量...";
        resultText.style.color = "#FF5722";
        uploadBtn.disabled = true; 
        
        setTimeout(() => {
            runModel(preview);
        }, 100); 
    };
    reader.readAsDataURL(file);
});

async function runModel(imgElement) {
    try {
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imgElement, 0, 0, 224, 224);
        const imgData = ctx.getImageData(0, 0, 224, 224).data;

        const floatData = new Float32Array(3 * 224 * 224);
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        for (let i = 0; i < 224 * 224; i++) {
            let r = imgData[i * 4 + 0] / 255.0;
            let g = imgData[i * 4 + 1] / 255.0;
            let b = imgData[i * 4 + 2] / 255.0;
            floatData[i] = (r - mean[0]) / std[0]; 
            floatData[224 * 224 + i] = (g - mean[1]) / std[1]; 
            floatData[2 * 224 * 224 + i] = (b - mean[2]) / std[2]; 
        }

        const tensor = new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
        const feeds = { [mySession.inputNames[0]]: tensor };
        const results = await mySession.run(feeds);

        const outputName = mySession.outputNames[0];
        let score = results[outputName].data[0];
        
        uploadBtn.disabled = false;
        resultText.innerText = `你的jj长度：${score.toFixed(2)} cm！`;

    } catch (e) {
        console.error(e);
        uploadBtn.disabled = false;
        resultText.innerText = "❌ 运算崩溃。";
    }
}
