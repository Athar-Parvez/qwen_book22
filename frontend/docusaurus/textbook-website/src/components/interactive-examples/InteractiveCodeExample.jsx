import React, { useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

// A component for interactive code examples with execution simulation
const InteractiveCodeExample = ({ 
  code, 
  language = 'python', 
  title = 'Code Example',
  description = '',
  output = 'Run the code to see output',
  allowRun = true
}) => {
  const [currentCode, setCurrentCode] = useState(code);
  const [currentOutput, setCurrentOutput] = useState(output);
  const [isRunning, setIsRunning] = useState(false);

  const handleRunCode = () => {
    if (!allowRun) return;
    
    setIsRunning(true);
    
    // Simulate code execution delay
    setTimeout(() => {
      // In a real implementation, this would connect to a backend service
      // to actually execute the code in a safe environment
      try {
        // This is a simulation - in a real implementation, we'd send the code
        // to a backend service that executes it safely
        let simulatedOutput = `Executed: ${language} code\n`;
        
        // Add some language-specific simulated output
        if (language === 'python') {
          simulatedOutput += simulatePythonOutput(currentCode);
        } else if (language === 'cpp' || language === 'c++') {
          simulatedOutput += simulateCppOutput(currentCode);
        } else if (language === 'bash' || language === 'shell') {
          simulatedOutput += simulateShellOutput(currentCode);
        } else {
          simulatedOutput += 'Code executed successfully\n';
        }
        
        setCurrentOutput(simulatedOutput);
      } catch (error) {
        setCurrentOutput(`Error: ${error.message}`);
      } finally {
        setIsRunning(false);
      }
    }, 1000); // 1 second delay to simulate processing
  };

  const handleResetCode = () => {
    setCurrentCode(code); // Reset to original code
    setCurrentOutput(output); // Reset to original output
  };

  const simulatePythonOutput = (pyCode) => {
    if (pyCode.includes('print(')) {
      // Extract content from print statements
      const printMatches = pyCode.match(/print\((.*?)\)/g);
      if (printMatches) {
        return printMatches.map(match => {
          const content = match.replace('print(', '').replace(')', '');
          return eval(`\`${content}\``); // Note: In a real implementation, never use eval() for user code
        }).join('\n');
      }
    }
    
    if (pyCode.includes('import')) {
      return 'Importing required libraries...\nReady to execute\n';
    }
    
    return 'Python code executed\n';
  };

  const simulateCppOutput = (cppCode) => {
    if (cppCode.includes('cout')) {
      // Extract content from cout statements
      const coutMatches = cppCode.match(/cout << (.*?);/g);
      if (coutMatches) {
        return coutMatches.map(match => 
          match.replace('cout << ', '').replace(';', '').trim()
        ).join('\n');
      }
    }
    
    return 'C++ code compiled and executed\n';
  };

  const simulateShellOutput = (shellCode) => {
    if (shellCode.includes('echo')) {
      const echoMatches = shellCode.match(/echo (.*?)(?:\n|$)/g);
      if (echoMatches) {
        return echoMatches.map(match => 
          match.replace('echo ', '').trim()
        ).join('\n');
      }
    }
    
    return 'Shell command executed\n';
  };

  return (
    <BrowserOnly>
      {() => (
        <div className="interactive-code-example">
          <div className="code-example-header">
            <h4>{title}</h4>
            {description && <p className="code-description">{description}</p>}
          </div>
          
          <div className="code-editor">
            <div className="editor-header">
              <div className="editor-tabs">
                <span className="active-tab">{language}</span>
              </div>
              <div className="editor-actions">
                {allowRun && (
                  <button 
                    className={`run-button ${isRunning ? 'running' : ''}`}
                    onClick={handleRunCode}
                    disabled={isRunning}
                  >
                    {isRunning ? 'Running...' : '▶ Run'}
                  </button>
                )}
                <button 
                  className="reset-button"
                  onClick={handleResetCode}
                >
                  ♺ Reset
                </button>
              </div>
            </div>
            
            <textarea
              value={currentCode}
              onChange={(e) => setCurrentCode(e.target.value)}
              className="code-textarea"
              spellCheck="false"
            />
          </div>
          
          <div className="code-output">
            <div className="output-header">
              <h5>Output</h5>
            </div>
            <pre className="output-content">
              {currentOutput}
            </pre>
          </div>
        </div>
      )}
    </BrowserOnly>
  );
};

export default InteractiveCodeExample;