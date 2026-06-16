let socket = null;

function connectWebSocket() {
    const wsUrl = `ws://${window.location.host}/ws`;
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        const displayTerminal = document.getElementById('monitor-terminal');
        displayTerminal.querySelector('#status-display').textContent = "[State]: System Idle";
        displayTerminal.querySelector('#status-display').className = "status-frame status-IDLE";
        displayTerminal.querySelector('#rationale-display').textContent = "WebSocket Connected. Awaiting telemetry load instruction execution...";
    };

    socket.onmessage = (event) => {
        const analyticalResult = JSON.parse(event.data);
        const displayTerminal = document.getElementById('monitor-terminal');
        
        if (analyticalResult.status === "success") {
            const statusDisplay = displayTerminal.querySelector('#status-display');
            statusDisplay.textContent = `[State]: ${analyticalResult.alert_level}`;
            statusDisplay.className = `status-frame status-${analyticalResult.alert_level}`;
            
            const rationaleDisplay = displayTerminal.querySelector('#rationale-display');
            rationaleDisplay.replaceChildren(); // clear
            
            const metricNode = document.createElement('div');
            const boldMetric = document.createElement('strong');
            boldMetric.textContent = "Risk Metric Scalar: ";
            metricNode.appendChild(boldMetric);
            metricNode.appendChild(document.createTextNode(analyticalResult.probability.toFixed(6)));
            
            const justNode = document.createElement('div');
            justNode.style.marginTop = "10px";
            const boldJust = document.createElement('strong');
            boldJust.textContent = "Linguistic Justification:";
            justNode.appendChild(boldJust);
            justNode.appendChild(document.createElement('br'));
            justNode.appendChild(document.createTextNode(analyticalResult.rationale));
            
            rationaleDisplay.appendChild(metricNode);
            rationaleDisplay.appendChild(justNode);
            
        } else {
            const statusDisplay = displayTerminal.querySelector('#status-display');
            statusDisplay.textContent = `[State]: Engine Failure`;
            statusDisplay.className = `status-frame status-CRITICAL`;
            
            const rationaleDisplay = displayTerminal.querySelector('#rationale-display');
            rationaleDisplay.textContent = `Reason: ${analyticalResult.detail}`;
        }
    };

    socket.onclose = () => {
        const displayTerminal = document.getElementById('monitor-terminal');
        displayTerminal.querySelector('#status-display').textContent = "[State]: Disconnected";
        displayTerminal.querySelector('#status-display').className = "status-frame status-CRITICAL";
        displayTerminal.querySelector('#rationale-display').textContent = "Connection lost. Reconnecting...";
        setTimeout(connectWebSocket, 5000);
    };
}

document.getElementById('execution-trigger').addEventListener('click', () => {
    const displayTerminal = document.getElementById('monitor-terminal');
    displayTerminal.querySelector('#status-display').textContent = "[State]: Synchronizing Tensors...";
    displayTerminal.querySelector('#status-display').className = "status-frame status-IDLE";
    displayTerminal.querySelector('#rationale-display').textContent = "Executing non-blocking framework verification routines.";

    const vectorPayload = {
        peep: parseFloat(document.getElementById('param-peep').value),
        pip: parseFloat(document.getElementById('param-pip').value),
        fio2: parseFloat(document.getElementById('param-fio2').value),
        hrv: parseFloat(document.getElementById('param-hrv').value),
        procalcitonin: parseFloat(document.getElementById('param-pct').value),
        p_f_ratio: parseFloat(document.getElementById('param-pf').value)
    };

    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(vectorPayload));
    } else {
        displayTerminal.querySelector('#status-display').textContent = "[State]: Ingress Interrupted";
        displayTerminal.querySelector('#status-display').className = "status-frame status-CRITICAL";
        displayTerminal.querySelector('#rationale-display').textContent = "WebSocket layer failed to establish connection to edge node.";
    }
});

// Initialize connection
connectWebSocket();
