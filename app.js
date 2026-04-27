const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const resultText = document.getElementById('result');

// 监听图片上传事件
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // 显示预览图
    const reader = new FileReader();
    reader.onload = (event) => {
        preview.src = event.target.result;
        preview.style.display = 'block';
        resultText.innerText = "⏳ 正在测量中...";
        
        // 等图片加载完再开始推理
        preview.onload = () => runModel(preview);
    };
    reader.readAsDataURL(file);
});

async function runModel(imgElement) {
    try {
        // 1. 加载模型。注意这里使用相对路径 './'，极其关键，有效防止 GitHub Pages 的 404 报错
        const session = await ort.InferenceSession.create('./my_aesthetic_scorer_web.onnx');

        // 2. 利用 Canvas 将图片暴力缩放为 224x224
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imgElement, 0, 0, 224, 224);
        const imgData = ctx.getImageData(0, 0, 224, 224).data;

        // 3. 魔法时刻：将像素数据转为 PyTorch 兼容的 Float32 张量格式 [1, 3, 224, 224]
        const floatData = new Float32Array(3 * 224 * 224);
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        for (let i = 0; i < 224 * 224; i++) {
            let r = imgData[i * 4 + 0] / 255.0;
            let g = imgData[i * 4 + 1] / 255.0;
            let b = imgData[i * 4 + 2] / 255.0;

            // R, G, B 分层排列 (Planar format)
            floatData[i] = (r - mean[0]) / std[0]; 
            floatData[224 * 224 + i] = (g - mean[1]) / std[1]; 
            floatData[2 * 224 * 224 + i] = (b - mean[2]) / std[2]; 
        }

        // 4. 创建 Tensor 并扔给模型
        const tensor = new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
        const feeds = { [session.inputNames[0]]: tensor };
        const results = await session.run(feeds);

        // 5. 提取并展示最后的分数！
        const outputName = session.outputNames[0];
        let score = results[outputName].data[0];
        resultText.innerText = `✨ 你的jj长度是：${score.toFixed(2)} cm！`;

    } catch (e) {
        console.error(e);
        resultText.innerText = "❌ 运行失败，看眼 F12 控制台是不是路径不对。";
    }
}