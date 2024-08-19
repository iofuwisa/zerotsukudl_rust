const path = require('path');
const CopyFilePlugin = require("copy-webpack-plugin");

module.exports = {
  entry: "./src/script/main.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "main.js",
  },
  plugins: [
    new CopyFilePlugin(['./src/index.html', 'nn.csv'])
  ],
};
