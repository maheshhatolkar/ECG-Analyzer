const { app, BrowserWindow } = require('electron')
const path = require('path')
const { spawn } = require('child_process')

let serverProcess = null

function startPythonServer() {
  // start the Flask server (web/app.py) using the venv python if available
  const python = process.env.PYTHON || 'python'
  const script = path.join(__dirname, '..', '..', 'web', 'app.py')
  serverProcess = spawn(python, [script], { stdio: 'inherit' })
  serverProcess.on('error', (err) => console.error('Server process error', err))
}

function stopPythonServer() {
  if (serverProcess) {
    try {
      serverProcess.kill()
    } catch (e) {
      console.warn('Failed to kill server process', e)
    }
    serverProcess = null
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  })
  // load the Flask web UI
  win.loadURL('http://127.0.0.1:5000')
}

app.whenReady().then(() => {
  startPythonServer()
  // give Flask a moment to start; in production you should wait for readiness
  setTimeout(() => {
    createWindow()
  }, 1200)

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  stopPythonServer()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
