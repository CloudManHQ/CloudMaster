const express = require('express');
const path = require('path');

const app = express();

// 提供静态文件
app.use(express.static(path.join(__dirname, 'public')));

// 提供src目录中的文件
app.use('/src', express.static(path.join(__dirname, 'src')));

// 提供node_modules中的文件
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')));

// 所有路由都返回index.html
app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});