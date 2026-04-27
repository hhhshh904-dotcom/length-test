const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const resultText = document.getElementById('result');
const statusText = document.getElementById('statusText');
const spinner = document.getElementById('loadingSpinner');
const uploadBtn = document.getElementById('customUploadBtn');

let mySession = null; // 全局变量，只加载一次模型！

// 网页一打开，就开始悄悄加载模型
async function initModel() {
    try {
        spinner.style.display = 'block';
        // 创建全局的推理会话
        mySession = await ort.InferenceSession.create('./my_aesthetic_scorer_web.onnx');
        
        // 加载成功后更新 UI
        spinner.style.display = 'none';
        statusText.style.display = 'none';
        uploadBtn.innerText = "📸 点击上传照片";
        uploadBtn.disabled = false; // 点亮上传按钮
        resultText.innerText = "尺子准备就绪！";
        resultText.style.color = "#4CAF50";
    } catch (e) {
        console.error(e);
        spinner.style.display = 'none';
        statusText.innerText = "❌ 加载失败，请刷新网页重试。";
        statusText.style.color = "red";
    }
}

// 启动初始化
initModel();

// 监听图片上传事件
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file || !mySession) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        preview.src = event.target.result;
        preview.style.display = 'inline-block';
        
        // 改变 UI 状态为计算中
        resultText.innerText = "🔥正在用尺子测量jj中...";
        resultText.style.color = "#FF5722";
        spinner.style.display = 'block';
        uploadBtn.disabled = true; // 计算时暂时禁用按钮
        
        // 【核心魔法】用 setTimeout 强行让出一点点时间，让浏览器先把上面那个转圈圈画出来，再去卡死算张量
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
        
        // 使用全局 session 进行推理
        const results = await mySession.run(feeds);

        const outputName = mySession.outputNames[0];
        let score = results[outputName].data[0];
        
        // 恢复 UI
        spinner.style.display = 'none';
        uploadBtn.disabled = false;
        resultText.innerText = `你的jj长度是：${score.toFixed(2)} cm！`;

    } catch (e) {
        console.error(e);
        spinner.style.display = 'none';
        uploadBtn.disabled = false;
        resultText.innerText = "❌ 运算崩溃。如果是切屏导致的，请刷新网页。";
    }
}
