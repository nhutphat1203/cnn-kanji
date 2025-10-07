const canvas = document.getElementById('demoCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Khởi tạo nền đen
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Thiết lập nét vẽ
ctx.strokeStyle = '#fff';
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.lineWidth = 10;

// Lấy các phần tử HTML
const charInput = document.getElementById('charInput');
const submitBtn = document.getElementById('submitBtn');
const resultBox = document.getElementById('resultBox');

// Bật/tắt nút đoán
charInput.addEventListener('input', () => {
  submitBtn.disabled = charInput.value.trim() === '';
});

// Xử lý vẽ
function draw(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();

  [lastX, lastY] = [x, y];
}

canvas.addEventListener('mousedown', (e) => {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top];
});
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

// Xóa canvas
function clearCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#fff';
  resultBox.innerHTML = '<p>Canvas đã được xóa.</p>';
}

// Gửi ảnh lên server
const canvasForm = document.getElementById('canvasForm');
canvasForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const charValue = charInput.value.trim();
  if (!charValue) {
    alert('Nhập ký tự trước khi submit!');
    return;
  }

  canvas.toBlob(async (blob) => {
    if (!blob) return;

    const formData = new FormData();
    const uniqueId = crypto.randomUUID();
    const fileName = `${charValue}_${uniqueId}.png`;

    formData.append('file', blob, fileName);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/imgreg', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        resultBox.innerHTML = `<p class="error">Lỗi: ${errData.error || 'Không xác định'}</p>`;
        return;
      }

      const data = await response.json();
      displayResults(data.predictions);
    } catch (err) {
      resultBox.innerHTML = `<p class="error">Lỗi khi gửi yêu cầu: ${err}</p>`;
    }
  }, 'image/png');
});

// Hiển thị kết quả
function displayResults(predictions) {
  if (!predictions || predictions.length === 0) {
    resultBox.innerHTML = '<p>Không có dự đoán nào.</p>';
    return;
  }

  const html = predictions
    .slice(0, 5)
    .map(([char, conf]) =>
      `<div class="result-item"><span>${char}</span><span>${(conf * 100).toFixed(2)}%</span></div>`
    )
    .join('');

  resultBox.innerHTML = html;
}
