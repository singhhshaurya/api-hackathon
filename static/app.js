/**
 * NAMASTE to ICD-11 TM2 Mapping API - Frontend Application
 */

const API_BASE = '';
const AUDIT_LOG_KEY = 'sparrow';

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', function () {
    console.log('NAMASTE Medical Interface initialized');
    initializeAuditLog();
    showSection('translate');
});

// Initialize audit log storage
function initializeAuditLog() {
    if (!localStorage.getItem(AUDIT_LOG_KEY)) {
        localStorage.setItem(AUDIT_LOG_KEY, JSON.stringify([]));
    }
}

// Navigation Management
function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });

    // Show target section
    const targetSection = document.getElementById(sectionName + '-section');
    if (targetSection) {
        targetSection.classList.add('active');
    }

    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });

    const activeLink = document.querySelector('a[onclick="showSection(\'' + sectionName + '\')"]');
    if (activeLink) {
        activeLink.classList.add('active');
    }
}

// Get authentication token
function getAuthToken() {
    // Try to get token from the UI input field first
    const tokenInput = document.getElementById('auth-token');
    if (tokenInput && tokenInput.value.trim()) {
        return tokenInput.value.trim();
    }

    // Fallback to localStorage if no token in UI
    let token = localStorage.getItem('auth_token');
    if (!token) {
        token = 'demo-token-' + Date.now();
        localStorage.setItem('auth_token', token);
    }
    return token;
}

// Generate request ID
function generateRequestId() {
    return 'req_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Get default headers for API calls
function getDefaultHeaders() {
    const headers = {
        'Content-Type': 'application/json',
        'X-Client-Version': '1.0.0',
        'X-Request-ID': generateRequestId()
    };

    const token = getAuthToken();
    if (token) {
        headers['Authorization'] = 'Bearer ' + token;
    }

    return headers;
}

// Make API call
async function makeApiCall(endpoint, options) {
    options = options || {};
    const url = API_BASE + endpoint;

    const config = {
        method: options.method || 'GET',
        headers: getDefaultHeaders()
    };

    if (options.body) {
        config.body = options.body;
    }

    if (options.headers) {
        Object.assign(config.headers, options.headers);
    }

    try {
        const response = await fetch(url, config);
        const data = await response.json();

        const result = {
            success: response.ok,
            status: response.status,
            data: data
        };

        logAuditEvent('api_call', {
            endpoint: endpoint,
            method: config.method,
            status: response.status,
            success: response.ok
        });

        return result;

    } catch (error) {
        const result = {
            success: false,
            status: 0,
            data: { error: 'network_error', message: error.message }
        };

        logAuditEvent('api_error', {
            endpoint: endpoint,
            error: error.message
        });

        return result;
    }
}

// Log audit events
function logAuditEvent(operation, details) {
    details = details || {};
    const timestamp = new Date().toISOString();
    const event = {
        id: generateRequestId(),
        timestamp: timestamp,
        operation: operation,
        details: details,
        session: {
            user_agent: navigator.userAgent,
            timestamp: timestamp
        }
    };

    const auditLog = JSON.parse(localStorage.getItem(AUDIT_LOG_KEY) || '[]');
    auditLog.unshift(event);

    // Keep only recent 1000 entries
    if (auditLog.length > 1000) {
        auditLog.splice(1000);
    }

    localStorage.setItem(AUDIT_LOG_KEY, JSON.stringify(auditLog));
    console.log('Audit Event:', operation, details);
}

// Display result in container
function displayResult(containerId, result) {
    const container = document.getElementById(containerId);
    container.classList.remove('success', 'error');
    container.classList.add('show');

    if (result.success) {
        container.classList.add('success');
        container.innerHTML = formatSuccessResult(result.data, result.status);
    } else {
        container.classList.add('error');
        container.innerHTML = formatErrorResult(result.data, result.status);
    }

    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Format successful result
function formatSuccessResult(data, status) {
    return '<div class="result-header">' +
        '<div class="result-status success">' +
        '<i class="status-icon">✓</i>' +
        '<span>Success (' + status + ')</span>' +
        '</div>' +
        '<div class="result-timestamp">' +
        formatTimestamp(new Date().toISOString()) +
        '</div>' +
        '</div>' +
        '<div class="result-content">' +
        formatDataForDisplay(data) +
        '</div>';
}

// Format error result
function formatErrorResult(data, status) {
    return '<div class="result-header">' +
        '<div class="result-status error">' +
        '<i class="status-icon">✗</i>' +
        '<span>Error (' + status + ')</span>' +
        '</div>' +
        '<div class="result-timestamp">' +
        formatTimestamp(new Date().toISOString()) +
        '</div>' +
        '</div>' +
        '<div class="result-content">' +
        '<div class="error-details">' +
        (data.message || 'An error occurred') +
        '</div>' +
        (data.error ? '<div class="error-type">Error Type: ' + data.error + '</div>' : '') +
        '</div>';
}

// Format data for display
function formatDataForDisplay(data) {
    if (typeof data === 'string') {
        return '<div class="response-text">' + escapeHtml(data) + '</div>';
    }

    if (Array.isArray(data)) {
        let html = '<div class="response-array">' +
            '<div class="array-header">Results (' + data.length + ' items)</div>' +
            '<div class="array-items">';

        for (let i = 0; i < data.length; i++) {
            html += '<div class="array-item">' +
                '<span class="item-index">' + (i + 1) + '.</span>' +
                '<span class="item-content">' + formatDataForDisplay(data[i]) + '</span>' +
                '</div>';
        }

        html += '</div></div>';
        return html;
    }

    if (typeof data === 'object' && data !== null) {
        let html = '<div class="response-object">';

        for (const key in data) {
            if (data.hasOwnProperty(key)) {
                html += '<div class="object-property">' +
                    '<span class="property-key">' + escapeHtml(key) + ':</span>' +
                    '<span class="property-value">' + formatDataForDisplay(data[key]) + '</span>' +
                    '</div>';
            }
        }

        html += '</div>';
        return html;
    }

    return '<div class="response-primitive">' + escapeHtml(String(data)) + '</div>';
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
}

// Show loading state
function showLoadingState(containerId) {
    const container = document.getElementById(containerId);
    container.classList.remove('success', 'error');
    container.classList.add('show');
    container.innerHTML = '<div class="result-header">' +
        '<div class="result-status loading">' +
        '<div class="loading-spinner"></div>' +
        '<span>Processing Request</span>' +
        '</div>' +
        '</div>' +
        '<div class="result-content">' +
        '<p>Please wait while your request is being processed.</p>' +
        '</div>';
}

// API Endpoint Functions
async function testHealth() {
    logAuditEvent('health_check_initiated');
    showLoadingState('health-result');

    const result = await makeApiCall('/health');
    displayResult('health-result', result);
}

async function testTranslate() {
    const namasteCode = document.getElementById('namaste-code').value.trim();

    if (!namasteCode) {
        displayResult('translate-result', {
            success: false,
            status: 400,
            data: { error: 'validation_error', message: 'Please enter a NAMASTE code' }
        });
        return;
    }

    logAuditEvent('translation_initiated', { namaste_code: namasteCode });
    showLoadingState('translate-result');

    const result = await makeApiCall('/translate/' + encodeURIComponent(namasteCode));
    displayResult('translate-result', result);
}

async function testCondition() {
    const namasteCode = document.getElementById('condition-code').value.trim();
    const subjectId = document.getElementById('subject-id').value.trim();
    const resourceId = document.getElementById('resource-id').value.trim();

    if (!namasteCode) {
        displayResult('condition-result', {
            success: false,
            status: 400,
            data: { error: 'validation_error', message: 'Please enter a NAMASTE code' }
        });
        return;
    }

    const requestData = {
        namaste_code: namasteCode
    };

    // Add optional fields if provided
    if (subjectId) {
        requestData.subject_id = subjectId;
    }
    if (resourceId) {
        requestData.resource_id = resourceId;
    }

    logAuditEvent('condition_creation_initiated', { namaste_code: namasteCode });
    showLoadingState('condition-result');

    const result = await makeApiCall('/condition', {
        method: 'POST',
        body: JSON.stringify(requestData)
    });

    displayResult('condition-result', result);
    logAuditEvent('condition_creation_completed', { success: result.success });
}

async function testCodeSystem() {
    const systemType = document.getElementById('system-type').value;

    logAuditEvent('codesystem_request_initiated', { system: systemType });
    showLoadingState('codesystem-result');

    const result = await makeApiCall('/codesystem/' + systemType);
    displayResult('codesystem-result', result);
}

async function viewAuditLogs() {
    logAuditEvent('audit_view_requested');
    showLoadingState('audit-result');

    try {
        const serverResult = await makeApiCall('/audit?limit=100');

        if (serverResult.success) {
            const localLogs = JSON.parse(localStorage.getItem(AUDIT_LOG_KEY) || '[]');
            displayResult('audit-result', {
                success: true,
                status: 200,
                data: {
                    server_logs: serverResult.data,
                    local_logs: localLogs.slice(0, 20)
                }
            });
        } else {
            const localLogs = JSON.parse(localStorage.getItem(AUDIT_LOG_KEY) || '[]');
            displayResult('audit-result', {
                success: true,
                status: 200,
                data: {
                    message: 'Server audit logs not available, showing local logs only',
                    local_logs: localLogs.slice(0, 50)
                }
            });
        }
    } catch (error) {
        displayResult('audit-result', {
            success: false,
            status: 500,
            data: { error: 'audit_error', message: 'Failed to retrieve audit logs' }
        });
    }
}

function clearAuditLogs() {
    if (confirm('Are you sure you want to clear all local audit logs? This action cannot be undone.')) {
        localStorage.setItem(AUDIT_LOG_KEY, JSON.stringify([]));
        logAuditEvent('audit_logs_cleared');

        const container = document.getElementById('audit-result');
        container.innerHTML = '<div class="result-header">' +
            '<div class="result-status success">' +
            '<i class="status-icon">✓</i>' +
            '<span>Cleared Successfully</span>' +
            '</div>' +
            '</div>' +
            '<div class="result-content">' +
            '<p>All local audit logs have been cleared.</p>' +
            '</div>';
        container.classList.add('show', 'success');
    }
}

// Test audit trail functionality
async function testAudit() {
    logAuditEvent('audit_test_initiated');
    showLoadingState('audit-result');

    const result = await makeApiCall('/audit', {
        method: 'GET'
    });

    displayResult('audit-result', result);
    logAuditEvent('audit_test_completed', { success: result.success });
}

console.log('NAMASTE Medical Interface loaded successfully');
